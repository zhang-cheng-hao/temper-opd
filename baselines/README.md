# Baselines

外部 baseline 统一下载到本目录。

默认下载目标：

```text
thunlp-opd        https://github.com/thunlp/OPD
tinker-cookbook   https://github.com/thinking-machines-lab/tinker-cookbook
flash-opd         https://github.com/china10s/flash-opd
opsd              https://github.com/siyan-zhao/OPSD
```

这些目录默认被父项目 `.gitignore` 忽略，避免直接 vendor 外部仓库。需要固定版本时，
再改成 git submodule 或在本文件记录 commit SHA。

## 当前下载版本

```text
thunlp-opd        1fd6cca846126af90d82ef122e8af261f59d2d37  archive 解压
tinker-cookbook   14374b5377e33f11d6dd057571037b11d4767322  git shallow clone
flash-opd         f2485a646dbddac997396cd36e36ee2e41d3e52e  git shallow clone
opsd              7448751f307a9cdbcc1246dd1565a1a605b443df  git shallow clone
```

`thunlp-opd` 使用 GitHub archive 下载，因为本机直接 `git clone` 多次卡在
`index-pack`。如需保留完整 Git metadata，后续可在网络和 I/O 稳定时改成 submodule。
