# This project incorporates modified code from [THU-BPM MarkLLM](https://github.com/THU-BPM/MarkLLM), licensed under the [Apache 2.0 License](http://www.apache.org/licenses/).Thanks for the contribution.
# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ===========================================
# color_scheme.py
# Description: Color scheme for visualization
# ===========================================

from matplotlib import pyplot as plt


class ColorScheme:
    """Color scheme for visualization."""

    def __init__(self, background_color='white', prefix_color='#6F6F6F') -> None:
        """
            Initialize the color scheme.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
        """
        self.background_color = background_color
        self.prefix_color = prefix_color
    
    def set_background_color(self, color) -> None:
        self.background_color = color
    
    def set_prefix_color(self, color) -> None:
        self.prefix_color = color
    
    def get_legend_height(self, font_size):
        return font_size  


class ColorSchemeForDiscreteVisualization(ColorScheme):
    """Color scheme for discrete visualization (KGW Family)."""

    def __init__(self, background_color='white', prefix_color='#6F6F6F', 
                 red_token_color='#CC7C71', green_token_color='#7AB656') -> None:
        """
            Initialize the color scheme for discrete visualization.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
                red_token_color (str): The color for red tokens.
                green_token_color (str): The color for green tokens.
        """
        super().__init__(background_color, prefix_color)
        self.red_token_color = red_token_color
        self.green_token_color = green_token_color
    
    def set_red_token_color(self, color) -> None:
        self.red_token_color = color
    
    def set_green_token_color(self, color)-> None:
        self.green_token_color = color
    
    def get_legend_items(self):
        return [
            ("Green Token", self.green_token_color),
            ("Red Token", self.red_token_color),
            ("Ignored", self.prefix_color)
        ]


class ColorSchemeForContinuousVisualization(ColorScheme):
    """Color scheme for continuous visualization (Christ Family)."""

    def __init__(self, background_color='white', prefix_color='#6F6F6F',
                 color_axis_name='plasma_r') -> None:
        """
            Initialize the color scheme for continuous visualization.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
                color_axis_name (str): The color axis name.
        """
        super().__init__(background_color, prefix_color)
        color_axis_name = color_axis_name
        self.color_axis = plt.get_cmap(color_axis_name)
    
    def set_color_axis(self, color_axis_name: str) -> None:
        self.color_axis = plt.get_cmap(color_axis_name)
    
    def get_color_from_axis(self, value: float) -> tuple:
        rgba_color = self.color_axis(value)
        rgba_color_int = tuple(int(255 * component) for component in rgba_color)
        return rgba_color_int[0], rgba_color_int[1], rgba_color_int[2]
    
    def get_legend_items(self):
        return [
            ("Prefix", self.prefix_color)
        ]

class ColorSchemeForMultiLevelVisualization(ColorScheme):
    """Color scheme for multi-level visualization (SynthID)."""

    def __init__(self,
                background_color: str = "#FFFFFF",
                prefix_color: str = "#000000",
                non_watermark_color: str = "#CCCCCC", 
                weak_watermark_color: str = "#f4895f",
                strong_watermark_color: str = "#de324c") -> None:
        """
            Initialize the color scheme for multi-level visualization.

            Parameters:
                background_color (str): The background color.
                prefix_color (str): The prefix color.
                non_watermark_color (str): The color for non-watermarked tokens.
                weak_watermark_color (str): The color for weak watermarked tokens.
                strong_watermark_color (str): The color for strong watermarked tokens.
        """
        super().__init__(background_color, prefix_color)
        self.non_watermark_color = non_watermark_color
        self.weak_watermark_color = weak_watermark_color
        self.strong_watermark_color = strong_watermark_color

    def get_legend_items(self):
        return [
            ("Non-watermarked", self.non_watermark_color),
            ("Weak watermark", self.weak_watermark_color),
            ("Strong watermark", self.strong_watermark_color)
        ]

class ColorSchemeForITSEdit(ColorSchemeForDiscreteVisualization):
    """Color scheme for ITSEdit visualization."""
    
    def __init__(self, 
                 background_color: str = "#FFFFFF",
                 prefix_color: str = "#ADB5BD",
                 watermark_color: str = "#FFD700", 
                 non_watermark_color: str = "#194a7a" 
                ) -> None:

        super().__init__(
            background_color=background_color,
            prefix_color=prefix_color,

             # Reuse the parameter name of the parent class but replace the semantics
            red_token_color=non_watermark_color, 
            green_token_color=watermark_color
        )

    @property
    def watermark_color(self) -> str:
        """Watermark token color alias. (To facilitate the use of DiscreteVisualizer)"""
        return self.green_token_color

    @property
    def non_watermark_color(self) -> str:
        """Unwatermark token color alias. (To facilitate the use of DiscreteVisualizer)"""
        return self.red_token_color

    def get_legend_items(self):
        return [
            ("Top-k token", self.watermark_color),
            ("General token", self.non_watermark_color),
            ("Prefix", self.prefix_color),
        ]
