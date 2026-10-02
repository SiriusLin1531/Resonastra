# Resonastra Release Verification Guide

[Project README](https://github.com/SiriusLin1531/Resonastra/blob/main/README.md) | [Quick Start](./quick-start.md) | [Compatibility](./compatibility.md) | **English** | [中文简体](../cn/release-verification.md)

Applies to: **Resonastra v1.0.0**

> [!NOTE]
> This guide helps you confirm that the Resonastra v1.0.0 package you downloaded is the official release package and that your local ZIP is byte-identical to the published release file.

---

## 1. What This Guide Covers

If you have already downloaded Resonastra and want to confirm that the file is correct and complete, use this guide to verify it.

This guide covers only:

- the official v1.0.0 release identity;
- the archive filename and size;
- calculating and comparing the ZIP SHA256;
- the relationship between the GitHub Release, the `v1.0.0` tag, and the current `main`;
- what to do when verification fails.

For installation, extraction, and first startup, continue with the [Quick Start](./quick-start.md). For Windows, GPU, and CUDA support boundaries, see [Compatibility](./compatibility.md).

---

## 2. Official v1.0.0 Release Identity

The official Resonastra v1.0.0 release information is:

| Item | Official Value |
| --- | --- |
| Version | `v1.0.0` |
| Archive | `Resonastra_v1.0.0.zip` |
| File size | `13,866,120,773 bytes` |
| SHA256 | `420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41` |
| GitHub Release | [Resonastra v1.0.0](https://github.com/SiriusLin1531/Resonastra/releases/tag/v1.0.0) |
| Download provider | Baidu Pan |
| Extraction code | `8848` |

Official Baidu Pan share:

```text
https://pan.baidu.com/s/1_344UD7sJci6IuBgtbmF6A?pwd=8848
```

The GitHub Release records the official release information and the `v1.0.0` tag. The full release package is distributed through the Baidu Pan share above rather than uploaded as a large GitHub Release asset.

> [!IMPORTANT]
> A working Baidu Pan share page only confirms that the download entry point is accessible. To confirm that your local ZIP is exactly the official release package, verify its **SHA256**.

---

## 3. Verify the Downloaded File

Verification is recommended after the download completes and before extraction.

If `Resonastra_v1.0.0.zip` is in your current PowerShell directory, run:

```powershell
(Get-FileHash .\Resonastra_v1.0.0.zip -Algorithm SHA256).Hash
```

The expected result is:

```text
420772A840FE496E175501085C20191C83B1433D366EA5BB060AF1C5DB093E41
```

Hexadecimal letter case does not affect the value, so the uppercase output above is equivalent to the lowercase SHA256 in the release record.

You can also use the Windows `certutil` command:

```bat
certutil -hashfile Resonastra_v1.0.0.zip SHA256
```

To optionally verify the file size, run:

```powershell
(Get-Item .\Resonastra_v1.0.0.zip).Length
```

The expected size is:

```text
13866120773
```

The filename and file size are useful secondary checks, but **SHA256 is the primary byte-identity check for the official release package**.

---

## 4. Interpret the Verification Result

If the calculated SHA256 exactly matches:

```text
420772a840fe496e175501085c20191c83b1433d366ea5bb060af1c5db093e41
```

then the ZIP you downloaded is byte-identical to the verified Resonastra v1.0.0 release archive, and you can continue with extraction and use.

If the SHA256 **does not match**:

1. do not treat the current ZIP as a verified official release package;
2. do not try to "repair" it by modifying files, adding missing files, or repackaging the archive;
3. download the package again from the distribution channel listed by the official GitHub Release;
4. calculate the SHA256 again after the new download completes.

A mismatched SHA256 may indicate an incomplete download, a corrupted file, or a different payload. The checksum result alone does not identify the exact cause.

---

## 5. GitHub Release, Tag, and Current main

For normal users, the relationship can be understood as:

```text
GitHub Release v1.0.0
  ↓
Records the official release and download information

v1.0.0 tag
  ↓
Fixes the historical source snapshot for that release

Current main
  ↓
May continue to receive maintained documentation and other living updates
```

The official Resonastra v1.0.0 tag is:

```text
v1.0.0
```

That tag points to the release source commit:

```text
f4d39c7c901d60d018d887f6e66c4357af2dd6f4
```

The `v1.0.0` tag is the immutable historical release snapshot. After the release, the public `main` branch may continue advancing through documentation and other maintenance updates, so **the current `main` does not need to remain equal to the v1.0.0 release commit**.

If you only want to verify that the downloaded package is correct, you do not need to compare the current `main` commit. Check the official GitHub Release and the ZIP SHA256 instead.

---

## 6. Next Steps

After verification passes:

1. extract `Resonastra_v1.0.0.zip`;
2. follow the [Quick Start](./quick-start.md) to run the environment check and first startup;
3. if you later encounter Windows, runtime, GPU, or CUDA compatibility issues, see [Compatibility](./compatibility.md).

If verification does not pass, download the package again and verify the SHA256 before continuing.
