#!/usr/bin/env node
// Read-only checks for explicitly selected context documents; never rewrites files.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { unified } from 'unified'
import remarkParse from 'remark-parse'

function walk(node, visit) {
  visit(node)
  for (const child of node.children ?? []) walk(child, visit)
}

export function checkContext(root, files) {
  const errors = [], warnings = [], paragraphs = new Map()
  for (const name of [...new Set(files)]) {
    const file = path.resolve(root, name)
    let source
    try { source = fs.readFileSync(file, 'utf8') } catch (error) {
      errors.push(`${name}: 无法读取 (${error.code})`)
      continue
    }
    const lines = source.trimEnd().split('\n')
    const normalized = name.replaceAll('\\', '/')
    const budget = normalized === 'global/AGENTS.md' || path.basename(name) === 'SKILL.md'
      ? 100 : /^(AGENTS|CLAUDE)\.md$/.test(path.basename(name)) ? 150 : null
    if (budget && lines.length > budget) warnings.push(`${name}: ${lines.length} 行,建议预算 ${budget} 行`)
    // Parse Markdown, so sample links inside code fences are not treated as references.
    const tree = unified().use(remarkParse).parse(source)
    walk(tree, node => {
      if (['link', 'image', 'definition'].includes(node.type)) {
        const url = node.url
        if (!url || /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(url)) return
        let target
        try { target = decodeURIComponent(url.split(/[?#]/)[0]) } catch {
          errors.push(`${name}:${node.position.start.line}: 链接编码无效`)
          return
        }
        if (target && !fs.existsSync(path.resolve(path.dirname(file), target))) {
          errors.push(`${name}:${node.position.start.line}: 断链 ${url}`)
        }
      }
      if (node.type !== 'paragraph') return
      const raw = source.slice(node.position.start.offset, node.position.end.offset)
      if (budget && raw.split('\n').some(line => [...line].length > 400)) {
        warnings.push(`${name}:${node.position.start.line}: 正文单行超过 400 字符,检查是否堆积或挤行`)
      }
      const key = raw.replace(/\s+/g, ' ').trim()
      if ([...key].length < 140) return
      const location = `${name}:${node.position.start.line}`
      if (paragraphs.has(key)) warnings.push(`${location}: 与 ${paragraphs.get(key)} 存在相同长段落`)
      else paragraphs.set(key, location)
    })
  }
  return { errors, warnings }
}

function main(args) {
  let root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
  if (args[0] === '--root') {
    if (!args[1]) throw new Error('--root 需要目录')
    root = path.resolve(args[1]); args = args.slice(2)
  }
  if (args.some(arg => arg.startsWith('--'))) throw new Error('用法: context-check.mjs [--root DIR] [FILE ...]')
  const files = args.length ? args : ['AGENTS.md', 'global/AGENTS.md', 'global/skills/maintain-project-context/SKILL.md']
  const { errors, warnings } = checkContext(root, files)
  for (const warning of warnings) console.log(`提示: ${warning}`)
  for (const error of errors) console.error(`错误: ${error}`)
  console.log(`上下文检查: ${files.length} 个文件,${errors.length} 个错误,${warnings.length} 个提示 (只读)`)
  process.exitCode = errors.length ? 1 : 0
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(process.argv.slice(2)) } catch (error) { console.error(error.message); process.exitCode = 2 }
}
