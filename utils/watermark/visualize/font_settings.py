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

# ============================================
# font_settings.py
# Description: Font settings for visualization
# ============================================

import os
from PIL import ImageFont


script_dir = os.path.dirname(__file__)
font_path = os.path.join(script_dir, 'font', 'Courier_New_Bold.ttf')

class FontSettings:
    """Font settings for visualization."""

    def __init__(self, font_path: str = font_path, font_size: int = 20) -> None:
        """
            Initialize the font settings.

            Parameters:
                font_path (str): The path to the font file.
                font_size (int): The font size.
        """
        self.font_path = font_path
        self.font_size = font_size
        self.font = ImageFont.truetype(self.font_path, self.font_size)