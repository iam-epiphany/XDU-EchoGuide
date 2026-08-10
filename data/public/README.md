# 公开信息数据目录

本目录存放西电校园**结构化公开信息**，由维护者填充真实数据后系统自动加载
（启动时读取，无需重启即可生效的说明见下）。

## 数据文件与用途

| 文件 | 内容 | 对话示例 |
|------|------|----------|
| `shuttle_schedule.json` | 校车/班车时刻表（南↔北） | "下一班校车几点？" |
| `buildings.json` | 楼宇位置（楼名/别名/位置/用途） | "信远楼在哪？" |
| `venues.json` | 运动场馆（位置/开放时段/设施） | "体育馆几点关门？" |
| `library.json` | 图书馆（各馆/自习区开放时间） | "图书馆几点开门？" |
| `academic_policies.json` | 带来源、版本和适用范围的学业政策样例 | "转专业要满足哪些条件？" |
| `affairs_processes.json` | 校园卡、请假、证明与缓考的办事流程 | "校园卡补办要什么材料？" |
| `it_diagnostics.json` | 校园网、VPN、统一认证与教务登录诊断树 | "认证成功但网页打不开" |

## 字段说明

### shuttle_schedule.json

```json
{
  "routes": [
    {
      "name": "南校区→北校区",
      "direction": "南→北",
      "pickup": "南校区东门乘车点",
      "duration_min": 60,
      "departures": ["07:00", "07:30", "08:00"],
      "days": "weekdays",
      "note": "工作日班次较多，周末/节假日减少，以校车管理通知为准"
    }
  ]
}
```

字段说明：
- `departures`：发车时刻数组（HH:MM），系统自动计算"下一班"与剩余分钟
- `days`：可选，`weekdays`（周一至周五）/ `weekend`（周六周日）/ `all`（默认，每天）；
  工作日与双休班次不同时，拆成多条 route 并分别标注 days

### buildings.json

```json
[
  {
    "name": "信远楼",
    "aliases": ["信远"],
    "campus": "南校区",
    "location": "南校区中部，图书馆南侧",
    "description": "人文学院、外国语学院教学楼",
    "note": ""
  }
]
```

### venues.json

```json
[
  {
    "name": "南校区体育馆",
    "campus": "南校区",
    "location": "南校区东侧",
    "open_hours": "8:00-22:00",
    "facilities": ["篮球场", "羽毛球场", "健身房"],
    "note": "部分场地需预约"
  }
]
```

### library.json

```json
[
  {
    "name": "南校区图书馆",
    "campus": "南校区",
    "open_hours": "8:00-22:00",
    "note": "考试周延长开放，节假日以公告为准"
  }
]
```

## 使用说明

- 填好数据后重启服务生效（`docker compose restart echoguide`），或调用
  `POST /campus/reload` 热加载（见 API 文档）。
- 某文件缺失时，对应查询会回复"数据暂未录入"，系统其余功能不受影响。
- 每条办事、IT 和学业政策规则都必须声明 `source_url`、`updated_at` 与 `version`；
  无法核验的内容必须明确标记为演示范围，不能伪装成实时官方接口。
- 校车、楼宇、场馆和图书馆数据目前包含演示占位项，实际使用前应按最新官方通知核验。
