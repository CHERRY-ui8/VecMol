# Figure 1 流程图设计建议
## 简化版本（如果空间有限）

如果Figure 1空间有限，可以只展示核心流程：

```
[Molecule] → [Field] → [Encoder] → [Codes] → [Decoder] → [Field] → [Molecule]
     ↑                                                      ↓
     └─────────────── Training ────────────────────────────┘

[Noise] → [DDPM] → [Codes] → [Decoder] → [Field] → [Gradient Ascent] → [Molecule]
```

然后在Figure 2中详细展示各模块的架构细节。





