# Resonastra 版本校验指南

[项目 README](./README.md) | [快速开始](./quick-start.md) | [Compatibility](./compatibility.md) | [English](../en/release-verification.md) | **中文简体**

适用版本：**Resonastra v1.0.0**

> [!NOTE]
> 本指南用于确认你下载的 Resonastra v1.0.0 是否为正式发布包，以及下载到本地的 ZIP 是否与正式发布文件完全一致。

---

## 1. 本指南解决什么问题

如果你已经下载了 Resonastra，但希望确认文件是否正确、完整，可以使用本指南进行校验。

本指南只负责：

- 确认 v1.0.0 的正式发布身份；
- 核对发布包文件名与大小；
- 计算并比较 ZIP 的 SHA256；
- 说明 GitHub Release、`v1.0.0` tag 与当前 `main` 的关系；
- 说明校验失败时应如何处理。

安装、解压后的首次启动请继续查看 [快速开始](./quick-start.md)。Windows、GPU 与 CUDA 支持范围请查看 [Compatibility](./compatibility.md)。

---

## 2. v1.0.0 正式发布身份

Resonastra v1.0.0 的正式发布信息如下：

| 项目 | 正式值 |
| --- | --- |
| 版本 | `v1.0.0` |
| 发布包 | `Resonastra_v1.0.0.zip` |
| 文件大小 | `13,866,120,773 字节` |
| SHA256 | `420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41` |
| GitHub Release | [Resonastra v1.0.0](https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0) |
| 下载渠道 | 百度网盘 |
| 提取码 | `8848` |

正式百度网盘分享地址：

```text
https://pan.baidu.com/s/1_344UD7sJci6IuBgtbmF6A?pwd=8848
```

GitHub Release 负责记录正式版本信息和 `v1.0.0` tag；完整发行包通过上面的百度网盘分享提供，而不是作为 GitHub Release 的大型附件上传。

> [!IMPORTANT]
> 百度网盘分享页能够正常打开，只说明下载入口可访问。判断你本地 ZIP 是否与正式发布包完全一致，应以 **SHA256** 为准。

---

## 3. 校验下载文件

建议在下载完成后、解压之前进行校验。

假设 `Resonastra_v1.0.0.zip` 位于当前 PowerShell 目录，运行：

```powershell
(Get-FileHash .\Resonastra_v1.0.0.zip -Algorithm SHA256).Hash
```

正确结果应为：

```text
420772A840FE496E175501085C20191C83B1433D366EA5BB060AF1C5DB093E41
```

十六进制字母大小写不影响结果，因此上面的输出与正式发布记录中的小写 SHA256 是同一个值。

也可以使用 Windows 自带的 `certutil`：

```bat
certutil -hashfile Resonastra_v1.0.0.zip SHA256
```

如果还希望核对文件大小，可以运行：

```powershell
(Get-Item .\Resonastra_v1.0.0.zip).Length
```

正确大小应为：

```text
13866120773
```

文件名和大小可以作为辅助检查，但 **SHA256 才是正式发布包字节身份的主要校验依据**。

---

## 4. 如何判断校验结果

如果计算出的 SHA256 与下面的值完全一致：

```text
420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41
```

说明你下载到的 ZIP 与经过确认的 Resonastra v1.0.0 正式发布包在字节层面一致，可以继续解压和使用。

如果 SHA256 **不一致**：

1. 不要把当前 ZIP 视为已经校验通过的正式发布包；
2. 不要尝试通过修改、补文件或重新压缩来“修复”这个 ZIP；
3. 从正式 GitHub Release 所列出的下载渠道重新获取文件；
4. 下载完成后重新计算 SHA256。

SHA256 不一致可能意味着下载不完整、文件已损坏，或你实际拿到的是不同内容的文件。仅凭校验结果无法进一步判断具体原因。

---

## 5. GitHub Release、tag 与当前 main

对普通用户来说，可以把三者理解为：

```text
GitHub Release v1.0.0
  ↓
记录正式版本与下载信息

v1.0.0 tag
  ↓
固定该版本对应的历史源码快照

当前 main
  ↓
可以继续更新文档等持续维护内容
```

Resonastra v1.0.0 的正式 tag 是：

```text
v1.0.0
```

该 tag 指向的发布源码 commit 为：

```text
f4d39c7c901d60d018d887f6e66c4357af2dd6f4
```

`v1.0.0` 是不可变的历史发布快照。发布之后，公开仓库的 `main` 可以继续因为文档等后续维护而前进，因此**当前 `main` 不需要继续等于 v1.0.0 的发布 commit**。

如果你只是验证下载包是否正确，不需要比较当前 `main` commit；直接核对正式 GitHub Release 和 ZIP SHA256 即可。

---

## 6. 下一步

校验通过后，可以继续：

1. 解压 `Resonastra_v1.0.0.zip`；
2. 按 [快速开始](./quick-start.md) 完成环境检查和第一次启动；
3. 如果后续遇到 Windows、运行环境、GPU 或 CUDA 兼容性问题，查看 [Compatibility](./compatibility.md)。

如果校验未通过，请先重新下载并再次确认 SHA256，再进入后续使用流程。
