# 装一个 skill 或一条全局指令,让两边都能用

**现行做法**:先判范围:各项目共同的常驻约束进 `global/AGENTS.md`,跨项目可复用且触发明确的操作进 `global/skills/`;只在本仓库成立的进 `AGENTS.md` 与 `.claude/skills/`;某类或某个外部项目的留在那个项目里,开源项目优先用上游自带的 skill(TensorRT-LLM、sglang、pytorch 都自带);给开源项目做贡献都用得上的放 oss-kit 的 `_kit/skills/`,由 `wire-skills.sh` 提交到各实验仓的 `lab` 分支。再判载体:每轮都要生效的写常驻指令,遇到才用的写 skill。真源只有一份:本仓库的 `.agents/skills` 软链到 `.claude/skills`;`global/` 由 `scripts/link-global.sh` 链到 `~/.claude/CLAUDE.md`、`~/.codex/AGENTS.md`、`~/.claude/skills/`、`~/.agents/skills/`;安装后用 `bash scripts/link-global.sh --check` 只读核验,实际触发在新会话另验。外部 skill 先完整读、不执行其中脚本,按本仓库手册改写后再放进来。

**原则**:一份真源,两边读;全局层越薄越好,因为它在每个项目的每次会话里都占上下文。

**Claude 与 Codex**:Claude Code 读 `~/.claude/CLAUDE.md` 与 `~/.claude/skills/`,项目 skill 从启动目录往上找到仓库根的 `.claude/skills/`;Codex 读 `~/.codex/AGENTS.md` 与 `~/.agents/skills/`,项目 skill 从当前目录往上找 `.agents/skills/`。上游只带 `.claude/skills/` 的项目,Codex 看不到那些 skill。Claude Code 的云端会话与云端定时任务(routine)不读本机用户级目录。

**本仓库落实在**:`global/`、`scripts/link-global.sh`、`.claude/skills/`、`.agents/skills`(软链接)、`docs/12-范式库维护.md` 第五节

**来源**:<https://code.claude.com/docs/en/skills>、<https://learn.chatgpt.com/docs/build-skills> · 核实于 2026-09-30
