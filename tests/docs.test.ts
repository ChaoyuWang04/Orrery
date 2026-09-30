import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { spawnSync } from 'node:child_process'
import { afterEach, describe, expect, it } from 'vitest'
import { checkContext } from '../scripts/context-check.mjs'

const projectRoot = path.resolve(import.meta.dirname, '..')
const docsRoot = path.join(projectRoot, 'docs')

const contextFixtures: string[] = []
function contextFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'orrery-context-'))
  contextFixtures.push(root)
  return root
}
afterEach(() => {
  for (const root of contextFixtures.splice(0)) fs.rmSync(root, { recursive: true, force: true })
})

describe('上下文检查与全局分发', () => {
  it('解析真实 Markdown 链接,忽略代码示例与远程 URL,只读不改文件', () => {
    const root = contextFixture()
    fs.writeFileSync(path.join(root, '已有 文档.md'), '# 已有\n')
    const source = '# 入口\n[文件](已有%20文档.md#标题)\n[站点](https://example.org)\n```md\n[示例](missing.md)\n```\n'
    fs.writeFileSync(path.join(root, 'AGENTS.md'), source)
    expect(checkContext(root, ['AGENTS.md'])).toEqual({ errors: [], warnings: [] })
    fs.appendFileSync(path.join(root, 'AGENTS.md'), '\n[断链][a]\n\n[a]: absent.md\n')
    const before = fs.readFileSync(path.join(root, 'AGENTS.md'), 'utf8')
    const result = checkContext(root, ['AGENTS.md'])
    expect(result.errors).toHaveLength(1)
    expect(result.errors[0]).toContain('absent.md')
    expect(fs.readFileSync(path.join(root, 'AGENTS.md'), 'utf8')).toBe(before)
  })

  it('超限、挤行与重复仅提示,长实验记录没有入口预算', () => {
    const root = contextFixture()
    const paragraph = '必须保留可复现证据和适用前提。'.repeat(35)
    fs.writeFileSync(path.join(root, 'AGENTS.md'), `${'# 标题\n'.repeat(151)}\n${paragraph}\n\n${paragraph}\n`)
    const result = checkContext(root, ['AGENTS.md'])
    expect(result.errors).toEqual([])
    expect(result.warnings.some((message: string) => message.includes('预算'))).toBe(true)
    expect(result.warnings.some((message: string) => message.includes('挤行'))).toBe(true)
    expect(result.warnings.some((message: string) => message.includes('相同长段落'))).toBe(true)
    fs.writeFileSync(path.join(root, 'experiment.md'), '# 实验\n'.repeat(200))
    expect(checkContext(root, ['experiment.md'])).toEqual({ errors: [], warnings: [] })
    const cli = spawnSync(process.execPath, [path.join(projectRoot, 'scripts/context-check.mjs'), '--root', root, 'AGENTS.md'], { encoding: 'utf8' })
    expect(cli.status, cli.stderr).toBe(0)
  })

  it('缺文件、错误编码和断链返回非零', () => {
    const root = contextFixture()
    fs.writeFileSync(path.join(root, 'AGENTS.md'), '[错误](%ZZ.md)\n\n![图](lost.png)\n')
    expect(checkContext(root, ['missing.md', 'AGENTS.md']).errors).toHaveLength(3)
    const cli = spawnSync(process.execPath, [path.join(projectRoot, 'scripts/context-check.mjs'), '--root', root, 'AGENTS.md'], { encoding: 'utf8' })
    expect(cli.status).toBe(1)
  })

  it('实际默认入口不存在断链', () => {
    expect(checkContext(projectRoot, ['AGENTS.md', 'global/AGENTS.md', 'global/skills/maintain-project-context/SKILL.md']).errors).toEqual([])
  })

  it('分发检查只读,安装可重复,错误或断链会失败且不被检查修复', () => {
    const root = contextFixture()
    const run = (...args: string[]) => spawnSync('bash', [path.join(projectRoot, 'scripts/link-global.sh'), ...args], {
      encoding: 'utf8', env: { ...process.env, ORRERY_LINK_ROOT: root },
    })
    expect(run('--check').status).toBe(1)
    expect(fs.readdirSync(root)).toEqual([])
    fs.mkdirSync(path.join(root, '.codex'))
    fs.writeFileSync(path.join(root, '.codex/AGENTS.md'), 'original')
    expect(run().status).toBe(0)
    const backup = fs.readdirSync(path.join(root, '.codex')).find(name => name.startsWith('AGENTS.md.bak-'))!
    expect(fs.readFileSync(path.join(root, '.codex', backup), 'utf8')).toBe('original')
    expect(run('--check').status).toBe(0)
    expect(run().status).toBe(0)
    const target = path.join(root, '.agents/skills/maintain-project-context')
    fs.unlinkSync(target)
    fs.symlinkSync(path.join(root, 'nonexistent'), target)
    expect(run('--check').status).toBe(1)
    expect(fs.readlinkSync(target)).toBe(path.join(root, 'nonexistent'))
    expect(run('--unexpected').status).toBe(2)
  })
})

function activeMarkdownFiles(): string[] {
  const files = [
    path.join(projectRoot, 'README.md'),
    path.join(projectRoot, 'AGENTS.md'),
    path.join(projectRoot, 'CLAUDE.md'),
  ]

  const walk = (dir: string) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) walk(full)
      else if (entry.name.endsWith('.md')) files.push(full)
    }
  }

  walk(docsRoot)
  return files
}

describe('活动文档', () => {
  it('本地 Markdown 链接都指向存在的文件', () => {
    const broken: string[] = []

    for (const file of activeMarkdownFiles()) {
      const text = fs.readFileSync(file, 'utf8')
      for (const match of text.matchAll(/!?(?:\[[^\]]*\])\(([^)]+)\)/g)) {
        const raw = match[1].trim().replace(/^<|>$/g, '')
        if (/^(?:https?:|mailto:|#)/.test(raw) || raw.startsWith('/')) continue
        const target = decodeURIComponent(raw.split('#')[0])
        if (!target || fs.existsSync(path.resolve(path.dirname(file), target))) continue
        broken.push(`${path.relative(projectRoot, file)} -> ${raw}`)
      }
    }

    expect(broken).toEqual([])
  })

  // AGENTS.md 是唯一真源;CLAUDE.md 只导入它,两份手工副本迟早漂移
  it('CLAUDE.md 只导入 AGENTS.md,入口指向全景地图', () => {
    const agents = fs.readFileSync(path.join(projectRoot, 'AGENTS.md'), 'utf8')
    const claude = fs.readFileSync(path.join(projectRoot, 'CLAUDE.md'), 'utf8')

    expect(claude.trim()).toBe('@AGENTS.md')
    expect(agents).toContain('docs/00-START.md')
  })

  it('Codex 与 Claude 读同一份 skill', () => {
    const link = path.join(projectRoot, '.agents/skills')
    expect(fs.lstatSync(link).isSymbolicLink()).toBe(true)
    expect(fs.readlinkSync(link)).toBe('../.claude/skills')
  })

  // 全局守则与全局 skill 的真源在 global/,由 scripts/link-global.sh 链到用户级
  it('全局层真源齐全,skill 名与目录名一致', () => {
    expect(fs.existsSync(path.join(projectRoot, 'global/AGENTS.md'))).toBe(true)
    expect(fs.existsSync(path.join(projectRoot, 'scripts/link-global.sh'))).toBe(true)
    const skillsDir = path.join(projectRoot, 'global/skills')
    for (const name of fs.readdirSync(skillsDir)) {
      const skill = fs.readFileSync(path.join(skillsDir, name, 'SKILL.md'), 'utf8')
      expect(skill, `global/skills/${name}`).toMatch(new RegExp(`^---\\nname: ${name}\\n`))
    }
  })

  it('入口统一为七个功能模块,运行设施不算模块', () => {
    const entryFiles = ['README.md', 'AGENTS.md', 'CLAUDE.md', 'docs/00-START.md']
    const entry = entryFiles
      .map((file) => fs.readFileSync(path.join(projectRoot, file), 'utf8'))
      .join('\n')

    expect(entry).toContain('七个功能模块')
    expect(entry).not.toContain('五个功能模块')
    expect(entry).not.toContain('六个功能模块')
  })

  it('入口地图覆盖全部现行手册', () => {
    const start = fs.readFileSync(path.join(docsRoot, '00-START.md'), 'utf8')
    for (const manual of [
      '01-TASK.md',
      '02-题库导入流程.md',
      '03-题目写作规范.md',
      '04-知识库地图.md',
      '05-知识库写作契约.md',
      '06-开源解读流程.md',
      '07-LeetCode清单.md',
      '08-常驻服务.md',
      '09-日常维护.md',
      '10-材料解读流程.md',
      '11-模拟面试系统.md',
      '12-范式库维护.md',
      '13-周会纪要.md',
      '14-开源贡献.md',
      '15-对外展示.md',
    ]) {
      expect(start, `00-START 缺少 ${manual}`).toContain(manual)
    }
  })

  it('主页把报告解读作为开源项目旁边的独立入口', () => {
    const home = fs.readFileSync(path.join(projectRoot, 'app/page.tsx'), 'utf8')
    const opensourceAt = home.indexOf('href="/opensource"')
    const reportsAt = home.indexOf('href="/reports"')

    expect(opensourceAt).toBeGreaterThanOrEqual(0)
    expect(reportsAt).toBeGreaterThan(opensourceAt)
    expect(home.slice(opensourceAt, reportsAt)).not.toContain('href="/leetcode"')
    expect(home).toContain('报告解读')
  })

  it('主页把日常研读作为报告解读旁边的独立入口', () => {
    const home = fs.readFileSync(path.join(projectRoot, 'app/page.tsx'), 'utf8')
    const reportsAt = home.indexOf('href="/reports"')
    const readingsAt = home.indexOf('href="/readings"')

    expect(readingsAt).toBeGreaterThan(reportsAt)
    expect(home.slice(reportsAt, readingsAt)).not.toContain('href="/leetcode"')
    expect(home).toContain('日常研读')
  })

  it('报告解读动态路由不对 Next 参数重复解码', () => {
    const route = fs.readFileSync(
      path.join(projectRoot, 'app/reports/[company]/[report]/page.tsx'),
      'utf8',
    )

    expect(route).not.toContain('decodeURIComponent')
  })

  // 方向名全是中文,动态段是百分号编码送进来的,不解码会 404(实测)。/opensource 同理
  it('日常研读动态路由必须解码中文方向段', () => {
    const route = fs.readFileSync(
      path.join(projectRoot, 'app/readings/[topic]/[paper]/page.tsx'),
      'utf8',
    )

    expect(route).toContain('decodeURIComponent')
  })

  it('不再恢复已废弃的静态面试池说明', () => {
    const task = fs.readFileSync(path.join(docsRoot, '01-TASK.md'), 'utf8')
    const interview = fs.readFileSync(path.join(docsRoot, '11-模拟面试系统.md'), 'utf8')
    const active = `${task}\n${interview}`

    expect(active).not.toMatch(/94 道|365 条|93 项/)
  })
})
