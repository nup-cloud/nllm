# Security Architecture -- nllm

## Overview

nllm はIoTデバイスを制御するため、セキュリティは最優先事項です。
本ドキュメントでは多層防御アーキテクチャを説明します。

## Threat Model

### 主な脅威

| Threat | Impact | Mitigation |
|--------|--------|------------|
| プロンプトインジェクション | 不正コマンド実行 | 多層サニタイザー + ホワイトリスト |
| コマンドインジェクション | デバイス乗っ取り | ホワイトリスト制御 + パラメータ検証 |
| データ漏洩 | 個人情報流出 | 完全オフライン + ローカルストレージ |
| モデル改ざん | 悪意あるコマンド生成 | SHA-256 整合性検証 |
| 権限昇格 | 管理者権限の奪取 | RBAC + 最小権限原則 |

## Defense Layers

```
Layer 1: Input Sanitization
  └─ src/nllm/core/sanitizer.py
     ├─ Banned pattern detection (日英対応)
     ├─ Control character removal
     └─ Input length limiting

Layer 2: Command Whitelist
  └─ src/nllm/core/whitelist.py
     ├─ Domain-specific allowed commands
     ├─ Unknown commands blocked by default
     └─ Customizable per deployment

Layer 3: Parameter Validation
  └─ src/nllm/sensor/range_check.py
  └─ src/nllm/sensor/schema.py
     ├─ Value range enforcement
     ├─ Type validation
     └─ Anomaly detection

Layer 4: Safety Policy Enforcement
  └─ src/nllm/device/safety.py
  └─ src/nllm/device/drone.py
     ├─ Altitude/speed limits
     ├─ Battery threshold checks
     ├─ Geofence enforcement
     └─ GPS requirement validation

Layer 5: Human-in-the-Loop
  └─ src/nllm/planning/executor.py
     ├─ Approval required for dangerous actions
     ├─ Confirmation callback system
     └─ Audit logging of all decisions

Layer 6: Audit Trail
  └─ src/nllm/planning/memory.py
     ├─ All commands logged with timestamps
     ├─ All decisions recorded
     └─ Persistent local storage
```

## Offline-First Security

- モデル推論は完全ローカル実行
- センサーデータは外部送信しない
- 学習データはローカルストレージのみ
- OTAアップデートは署名検証必須

## Model Integrity

```bash
# モデルファイルの整合性検証
./models/verify.sh check
```

SHA-256ハッシュによるモデルファイルの改ざん検知を実装。

## Incident Response

1. 異常検知: AlertPipeline が自動検知
2. 自動対応: 緊急停止コマンドの自動実行
3. ログ保全: 全操作履歴の自動保存
4. 通知: 管理者への即座アラート
