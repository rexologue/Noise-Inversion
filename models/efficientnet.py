import math
from typing import Literal

import torch
from torch.nn import Linear, Conv2d, GroupNorm, SiLU, AdaptiveAvgPool2d, Identity, Sequential, Module

from models.se import SEModule
from models.utils import Flatten


class MBConv(Module):
    """
    MBConv – инверсный резидуальный блок с механикой Squeeze-and-Excitation (SE),
    используемый в EfficientNetV2. Блок может опционально использовать SE-модуль,
    если параметр `use_se` равен True. Если коэффициент расширения (expansion_ratio) равен 1,
    блок не расширяет число каналов.
    
    Аргументы:
        in_channels (int): Число входных каналов.
        output_channels (int): Число выходных каналов.
        expansion_ratio (int): Коэффициент расширения числа каналов.
        downsample (bool, optional): Если True, применяется даунсэмплинг (stride=2). По умолчанию False.
        use_se (bool, optional): Если True, к промежуточным операциям добавляется SE-модуль.
                                 Если False, используется Identity вместо SE. По умолчанию True.
    """
    def __init__(self, 
                 in_channels: int, 
                 output_channels: int, 
                 expansion_ratio: int, 
                 downsample: bool = False,
                 use_se: bool = True):
        
        super(MBConv, self).__init__()

        # Определяем страйд: 2 при даунсэмплинге, иначе 1.
        stride = 2 if downsample else 1
        # Определяем, требуется ли блок расширения каналов (если expansion_ratio != 1).
        apply_expansion = (expansion_ratio != 1)
        expanded_channels = expansion_ratio * in_channels

        # Блок расширения:
        # Свёртка 1x1, за которой следует GroupNorm и активация SiLU.
        expansion_block = Sequential(
            # Расширение числа каналов.
            Conv2d(in_channels, expanded_channels, kernel_size=1),
            GroupNorm(1, expanded_channels),  # Аналог LayerNorm для изображений.
            SiLU(inplace=True)
        )

        # Основной остаточный путь:
        # Если требуется, сначала применяется блок расширения.
        # Затем выполняется depthwise свёртка 3x3 с учетом stride и группировки,
        # GroupNorm, активация SiLU, опциональный SE-модуль (если use_se True),
        # и последняя свёртка 1x1 для проекции каналов, за которой следует GroupNorm.
        self.residual = Sequential(
            expansion_block if apply_expansion else Identity(),
            
            # Depthwise свёртка с соответствующим даунсэмплингом.
            Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, groups=expanded_channels),
            GroupNorm(1, expanded_channels),  
            SiLU(inplace=True),

            # SE-модуль применяется, если use_se True, иначе Identity.
            SEModule(expanded_channels, reduction=(4 * expansion_ratio)) if use_se else Identity(),

            # Проекция: свёртка 1x1 для сжатия числа каналов до output_channels.
            Conv2d(expanded_channels, output_channels, kernel_size=1),
            GroupNorm(1, output_channels)
        )

        # Shortcut (прямой путь) используется только, если выполняется Identity соединение:
        # shortcut применяется только если не происходит даунсэмплинг и число входных
        # каналов равно числу выходных.
        self.apply_shortcut = not (downsample or in_channels != output_channels)

        
    def forward(self, x):
        """
        Прямой проход через блок MBConv.
        Вычисляется остаточный путь, к которому условно прибавляется
        исходный вход (shortcut), если размеры входа и выхода совпадают.
        
        Аргументы:
            x (torch.Tensor): Входной тензор.
        
        Возвращает:
            torch.Tensor: Выходной тензор после объединения residual и shortcut (при совпадении размерностей).
        """
        out = self.residual(x)

        if self.apply_shortcut:
            out += x

        return out
    

class FusedMBConv(Module):
    """
    FusedMBConv – модифицированный вариант MBConv, в котором объединяются операции расширения
    и depthwise-свёртки. В блоке используется свёртка 3×3, которая сразу расширяет число каналов,
    за ней следует GroupNorm, активация SiLU, опциональный SE-модуль (если use_se True) и,
    если требуется изменение размерности, свёртка 1×1 с GroupNorm для проекции.
    
    Аргументы:
        in_channels (int): Число входных каналов.
        output_channels (int): Число выходных каналов.
        expansion_ratio (int): Коэффициент расширения числа каналов.
        downsample (bool, optional): Если True, выполняется даунсэмплинг (stride=2). По умолчанию False.
        use_se (bool, optional): Если True, к операции добавляется SE-модуль; иначе используется Identity. По умолчанию True.
    """
    def __init__(self,
                 in_channels: int, 
                 output_channels: int, 
                 expansion_ratio: int, 
                 downsample: bool = False,
                 use_se: bool = True):
        
        super(FusedMBConv, self).__init__()

        # Определяем stride: 2 при даунсэмплинге, иначе 1.
        stride = 2 if downsample else 1
        expanded_channels = expansion_ratio * in_channels

        # Определяем, требуется ли операция проекции (если expansion_ratio != 1 или число каналов различается).
        apply_reduction = not (expansion_ratio == 1 and in_channels == output_channels)

        # Проекционный блок для сжатия каналов.
        reduction_block = Sequential(
            Conv2d(expanded_channels, output_channels, kernel_size=1, bias=False),
            GroupNorm(1, output_channels)
        )

        # Основной остаточный путь:
        # Объединенная свёртка 3x3, которая сразу расширяет число каналов,
        # GroupNorm и SiLU, затем SE-модуль (если use_se True) и, если требуется, проекция с помощью reduction_block.
        self.residual = Sequential(
            Conv2d(in_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            GroupNorm(1, expanded_channels),  
            SiLU(inplace=True),

            # SE-модуль или Identity, в зависимости от параметра use_se.
            SEModule(expanded_channels, reduction=(4 * expansion_ratio)) if use_se else Identity(),

            # Если необходимо, выполняется свёртка 1x1 для проекции выходных каналов.
            reduction_block if apply_reduction else Identity()
        )

        # Shortcut используется только, если не происходит даунсэмплинг и число входных каналов совпадает с числом выходных.
        self.apply_shortcut = not (downsample or in_channels != output_channels)


    def forward(self, x):
        """
        Прямой проход через блок FusedMBConv.
        Вычисляется остаточный путь, который суммируется с входом (shortcut)
        при совпадении размерностей.
        
        Аргументы:
            x (torch.Tensor): Входной тензор.
        
        Возвращает:
            torch.Tensor: Сумма выхода остаточного пути и shortcut (если применимо).
        """
        out = self.residual(x)

        if self.apply_shortcut:
            out += x
            
        return out


class EfficientNetV2(Module):
    """
    EfficientNetV2 – основная архитектура сети для извлечения признаков из изображений,
    которая строится из последовательных стадий, содержащих блоки MBConv или FusedMBConv.
    
    Особенности:
      - Используется stem-блок, состоящий из свёртки 3x3 с даунсэмплингом, GroupNorm и активации SiLU.
      - Первые несколько стадий используют FusedMBConv (для более эффективного вычисления на ранних слоях),
        а поздние стадии – MBConv.
      - Для формирования каждой стадии используются списки, задающие число каналов, количество блоков и коэффициенты расширения.
      - Финальный блок состоит из 1x1 свёртки, GroupNorm, активации SiLU, глобального усредняющего пуллинга и flatten.
      - Все слои проходят инициализацию весов с использованием Kaiming Normal (для Conv2d), стандартной инициализации для GroupNorm и Linear.
    
    Аргументы:
        arch (Literal['s', 'm', 'l']): Вариант модели – 's' для Small, 'm' для Medium, 'l' для Large.
    """
    def __init__(self, arch: Literal['s', 'm', 'l']):
        super(EfficientNetV2, self).__init__()

        # Стем-блок: свёртка 3x3 (stride=2), GroupNorm и SiLU.
        stem_channels = 32 if arch == 'l' else 24
        self.output_channels = 1280
        self.stem = Sequential(
            Conv2d(3, stem_channels, kernel_size=3, stride=2, bias=False),
            GroupNorm(1, stem_channels),
            SiLU(inplace=True)
        )

        # Количество стадий, где применяются FusedMBConv-блоки (например, в первых fused_stages_amount стадиях).
        fused_stages_amount  = 3
        # Получаем списки параметров для каждой стадии.
        channels_per_stage, blocks_per_stage, expansions_per_stage = self._get_arch_lists(arch)

        stages = []

        # Построение стадий: для первой блока каждой стадии число входных каналов берется из предыдущего этапа.
        for i in range(len(blocks_per_stage)):
            stages.extend(
                self._build_stage(
                    blocks_per_stage[i],
                    channels_per_stage[i - 1],  # для i == 0 берется последний элемент списка (каналы стем-выхода)
                    channels_per_stage[i],
                    expansions_per_stage[i],
                    (i < fused_stages_amount)  # Если i меньше fused_stages_amount, используем FusedMBConv.
                )
            )

        self.core = Sequential(*stages)

        # Финальный блок: 1x1 свёртка для проекции, GroupNorm, SiLU, глобальный усредняющий пуллинг и flatten.
        self.final_block = Sequential(
            Conv2d(channels_per_stage[-2], self.output_channels, kernel_size=1, bias=False),
            GroupNorm(1, self.output_channels),
            SiLU(inplace=True),
            AdaptiveAvgPool2d(1),
            Flatten()
        )

        # Инициализация весов для всех слоёв.
        for m in self.modules():
            if isinstance(m, Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, GroupNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                torch.nn.init.uniform_(m.weight, -init_range, init_range)
                torch.nn.init.zeros_(m.bias)


    def _build_stage(self, 
                     blocks_amount: int, 
                     in_channels: int, 
                     output_channels: int, 
                     expansion_ratio: int,
                     fused: bool = False) -> list:
        """
        Строит одну стадию сети, состоящую из последовательности блоков,
        которые могут быть либо MBConv, либо FusedMBConv.
    
        Аргументы:
            blocks_amount (int): Количество блоков в стадии.
            in_channels (int): Число входных каналов для первого блока данной стадии.
            output_channels (int): Число выходных каналов для всей стадии.
            expansion_ratio (int): Коэффициент расширения для блоков.
            fused (bool): Если True, в стадии используются FusedMBConv блоки,
                          иначе используются стандартные MBConv блоки.
    
        Возвращает:
            list: Список блоков, формирующих стадию.
        """
        blocks = []
        for i in range(blocks_amount):
            # В первом блоке используется указанное in_channels, а далее входное число каналов равно output_channels.
            block_in_channels = in_channels if i == 0 else output_channels
            if fused:
                blocks.append(
                    FusedMBConv(block_in_channels, output_channels, expansion_ratio, (i == 0), False)
                )
            else:
                blocks.append(
                    MBConv(block_in_channels, output_channels, expansion_ratio, (i == 0), True)
                )
        return blocks
    

    def forward(self, x):
        """
        Выполняет прямой проход через модель EfficientNetV2.
        Сначала применяется стем-блок, затем ядро (core) составленное из стадий,
        после чего финальный блок формирует вектор признаков.
    
        Аргументы:
            x (torch.Tensor): Входной тензор изображений.
    
        Возвращает:
            torch.Tensor: Выходной вектор признаков, полученный после финального блока.
        """
        out = self.stem(x)
        out = self.core(out)
        out = self.final_block(out)
        return out
    

    def _get_arch_lists(self, arch: Literal['s', 'm', 'l']):
        """
        Возвращает три списка параметров на основе выбранного варианта модели:
          - channels_per_stage: число выходных каналов для каждой стадии.
          - blocks_per_stage: количество блоков в каждой стадии.
          - expansions_per_stage: коэффициенты расширения для каждого блока.
    
        Аргументы:
            arch (Literal['s', 'm', 'l']): Вариант модели:
                                           's' (Small), 'm' (Medium) или 'l' (Large).
    
        Возвращает:
            tuple: Кортеж из трёх списков (channels_per_stage, blocks_per_stage, expansions_per_stage).
    
        Замечание:
            Для 's' и 'm' используется stem_channels равное 24, а для 'l' — 32.
        """
        if arch == 's':
            channels_per_stage   = [24, 48, 64, 128, 160, 256, 24]
            blocks_per_stage     = [2, 4, 4, 6, 9, 15]
            expansions_per_stage = [1, 4, 4, 4, 6, 6]

        if arch == 'm':
            channels_per_stage   = [24, 48, 80, 160, 176, 304, 512, 24]
            blocks_per_stage     = [3, 5, 5, 7, 14, 18, 5]
            expansions_per_stage = [1, 4, 4, 4, 6, 6, 6]

        if arch == 'l':
            channels_per_stage   = [32, 64, 96, 192, 224, 384, 640, 32]
            blocks_per_stage     = [4, 7, 7, 10, 19, 25, 7]
            expansions_per_stage = [1, 4, 4, 4, 6, 6, 6]

        return (
            channels_per_stage,
            blocks_per_stage,
            expansions_per_stage,
        )
                

class EfficientNetV2Classifier(Module):
    """
    Классификатор на базе EfficientNetV2.
    Сеть EfficientNetV2 используется для извлечения признаков,
    после чего линейный слой формирует логиты для классификации.
    
    Аргументы:
        arch (Literal['s', 'm', 'l']): Вариант модели EfficientNetV2 ('s', 'm' или 'l').
        num_classes (int): Количество классов для классификации.
    """
    def __init__(self, arch: Literal['s', 'm', 'l'], num_classes: int):
        super(EfficientNetV2Classifier, self).__init__()

        self.efficientnet = EfficientNetV2(arch)
        self.classifier = Linear(1280, num_classes)

    def forward(self, x):
        """
        Прямой проход через классификатор.
        Сначала извлекаются признаки с помощью EfficientNetV2,
        затем применяется линейный слой для получения логитов (без активации).
    
        Аргументы:
            x (torch.Tensor): Входной тензор изображений.
    
        Возвращает:
            torch.Tensor: Логиты для каждого класса.
        """
        return self.classifier(self.efficientnet(x))

