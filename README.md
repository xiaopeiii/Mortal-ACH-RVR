# Mortal - Feature Update Summary

English is the primary language in this README. Chinese summary is included below.

## Overview
This repository provides a focused summary of recent Mortal-side feature work.
The update direction is centered on ACH + RVR training and stricter data alignment.

## Key Updates (EN)
1. ACH + RVR three-stage training pipeline
- Stage 1: Offline Teacher (privileged learning)
- Stage 2: Offline Student Distillation
- Stage 3: Online Strict ACH+RVR

2. Strict alignment audit before online training
- Added strict audit flow for trace/old-logp consistency
- Fail-fast policy for missing trace, missing old logp, and ambiguous mappings

3. Unified ACH+RVR model heads
- PolicyYHead / ValueHead / RelativeValueHead / ExpectedRewardNet
- Joint optimization for policy, value, relative value, and expected reward

4. Reproducible evaluation workflow
- Added fair comparison scripts for 1v3 and 2v2 settings
- Supports policy-vs-value head comparisons with fixed seeds

5. Engineering and config improvements
- Multi-profile config templates for different hardware scenarios
- Added a clear playbook for stage-based training/evaluation

## Attribution
- Original Mortal project author: Equim-chan
- Upstream source: https://github.com/Equim-chan/Mortal

## 中文摘要（辅）
1. 主要改动方向是 ACH+RVR 三阶段训练链路。
2. 加入 strict 对齐审计，训练前先做 trace/old_logp 质量检查。
3. 统一策略头、价值头、相对价值头、期望奖励网络的联合优化。
4. 补充可复现评测脚本和分阶段实验流程。

## Sensitive Data Notice
No personal secrets, local absolute paths, tokens, or private runtime logs are included in this repository.


<!--
Original upstream README is commented out below for reference.

<p align="center">
  <img src="https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/logo.png" width="550" />
</p>

# Mortal
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Equim-chan/Mortal/libriichi.yml?branch=main)](https://github.com/Equim-chan/Mortal/actions/workflows/libriichi.yml)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Equim-chan/Mortal/docs.yml?branch=main&label=docs)](https://mortal.ekyu.moe)
[![dependency status](https://deps.rs/repo/github/Equim-chan/Mortal/status.svg)](https://deps.rs/repo/github/Equim-chan/Mortal)
![GitHub top language](https://img.shields.io/github/languages/top/Equim-chan/Mortal)
![Lines of code](https://www.aschey.tech/tokei/github/Equim-chan/Mortal)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/Equim-chan/Mortal)
[![license](https://img.shields.io/github/license/Equim-chan/Mortal)](https://github.com/Equim-chan/Mortal/blob/main/LICENSE)

[![Donate](https://img.shields.io/badge/Donate-%E2%9D%A4%EF%B8%8F-blue?style=social)](https://mortal.ekyu.moe/donate.html)

Mortal ([鍑″か](https://www.mdbg.net/chinese/dictionary?wdqb=%E5%87%A1%E5%A4%AB)) is a free and open source AI for Japanese mahjong, powered by deep reinforcement learning.

Read the [**Documentation**](https://mortal.ekyu.moe) for everything about this work.

## Okay cool now give me the weights!
Read [this post](https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56) for details regarding this topic.

## License
### Code
[![AGPL-3.0-or-later](https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/agpl.png)](https://github.com/Equim-chan/Mortal/blob/main/LICENSE)

Copyright (C) 2021-2022 Equim

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

### Logo and Other Assets
[![CC BY-SA 4.0](https://github.com/Equim-chan/Mortal/raw/main/docs/src/assets/by-sa.png)](https://creativecommons.org/licenses/by-sa/4.0/)

-->

