---
name: report-renovation
description: 成批翻新 reports/ 或 readings/ 里「待重审重写」的已发布解读时用。包含批次编排、子 agent 派发模板、主会话验收清单、隔离验证、执笔记录的安全更新脚本与只提交本线路径的做法。规则本身在 docs/10 第六节,这里只放操作细节;平时写新解读不用读。
---

# 成批翻新已发布解读

规则与标准以 `docs/10-材料解读流程.md` 第六节「翻新已发布稿」为准(对照标杆收一版、旧稿一律待核、按配图条补图、删过程台账)。本 skill 只写怎么成批地做。

## 一、选篇与编排

- 待翻新清单:执笔记录里状态为「待重审重写」、且正文存在的篇目,按 `reports/index.md`(或 `readings/index.md`)该方向的行序取
- 一批 6 篇,3 个子 agent 各两篇;方向只剩 7–8 篇时可一次开 4 个 agent 收尾
- 搭配:长原件(60 页以上)配短原件;同一系列前后代交给同一个 agent;网页原件与 PDF 搭配
- 派发前看一眼 `git status`:别的线程正在改的篇(同目录兄弟篇、被本批引用的篇)写进派发说明「不要读来当依据、不要改」
- 核一遍原件在不在:`ls papers/<公司>/<材料>.*`、`pdfinfo` 看页数;网页原件在索引「一句话」里有 URL

## 二、派发模板

每个子 agent 一份,按篇替换尖括号。体裁:模型报告对照 DeepSeek-V2,其他基模技术对照 DeepSeekMoE。

```text
你负责翻新本仓库(<仓库绝对路径>)<报告解读|日常研读>库里的两篇已发布稿。这是主会话派发的已确认任务,直接做,不出意图卡、不向维护者提问。全程中文。先写完一篇再写下一篇。

## 你的两篇(只准改这两篇及其配图目录)
1. `<库>/<目录>/<材料>.md` ← 原件 `<原件路径>`(<页数> 页)。配图目录 `public/<库>/<slug>/`(新建)。<这一篇的特别交代:篇幅、原件性质、不要碰的兄弟篇>
2. ……
两篇都对照标杆 `reports/DeepSeek/<DeepSeek-V2|DeepSeekMoE>.md`(原件 `papers/DeepSeek/<同名>.pdf`)。

## 开工前必读
docs/10 全文(重点第一节配图条、第三节、第四节、第六节、第七节);docs/06 第五节「图」;docs/09「踩过的雷」;标杆全文并对照其原件翻几页。

## 翻新口径(按 10 第六节)
- 旧稿一律视为待核。常见错误:编造的成绩与例子、读错的图表数值(相邻两根柱、带不带工具、两张表混引)、把作者观察当实验结论、混进别的报告的内容、页码与首发日写错。按第三节重跑版本核验、首发日、报告地图,对照原件重核全部数字、页码、图表与结论,核不上的改正或删去
- 单页取文用 `pdftotext -layout -f N -l N`;页码以 pdfinfo 的页序为准
- 官方有更新版本:下载到 scratchpad 的 newpdf/ 并 pdfinfo 验证,以新版写稿,不覆盖 papers/,报告里给 cp 命令
- 首发日按第一节口径复核(HF createdAt 与权重预置上传、第三方平台登记都不算),有官方证据证明错了才改
- 网页原件:重新访问、记访问日期;整页文本加逐张读图;旧稿的成绩必须在现页面、存档或模型卡里找到出处,找不到就删
- 按配图条配关键 figure,图在前、字配合;不支撑叙事的图不配
- 「本文依据」只写最终那一版原件;删复述、过程台账、写作计划句、检索手段当论据的句子
- 正文里不写任何给我们自己的待办(如「知识库可以跟进的点」),只写进交付报告
- 全角标点、「」引号、加粗不贴标点、区间用 en dash、Mermaid 标签含括号或 @ 加双引号、跨篇纯文本篇名、公式只用 $ / $$
- 不改索引、执笔记录、knowledge/ 与任何其他文件;不 git add、不提交;共享 scratchpad 里用自建子目录

## 自查
npm test(失败的若是别人的文件注明即可);新 SVG 跑 node scripts/svg-check.mjs public/<库>/<slug>/

## 交付报告(两篇分开)
1 改了什么(新结构、配图文件名与对应原文图号页码、删了什么;旧稿被更正或删掉的事实错误逐条列出)
2 版本与首发日核验结论(有新版给 cp 命令;改首发日给官方证据)
3 重要数字与页码抽样 5 条
4 外部补充、未公开或无法核实的缺口;只能靠猜的地方
5 知识库可能要跟进的点
6 自查结果
```

子 agent 中途断连(API 或网络错误):用 SendMessage 让它从断点续作,先看自家文件现状再往下写,不重派。

## 三、主会话验收(每篇)

子 agent 最查不出的是事实与图,验收力气花在这里:

1. **抽样复核**:把交付报告的 5 条抽样逐条在原件里重跑,`pdftotext -layout -f N -l N <pdf> - | grep -E '<数字>'`;页上找不到就全文逐页搜,确认页码
2. **旧稿更正项挑两三条核**:尤其是「旧稿说原件自相矛盾」「排名第几」「谁赢谁」这类,对着表逐行数
3. **图逐格读**:改动最大的那张图用 `pdftoppm -f N -l N -r 110 -png` 渲染后看原图,再看新画的 SVG(浏览器打开 `file://` 路径);网页原件的图直接下载或在浏览器里读
4. **网页原件的表格数字**:页面多是 JS 渲染,WebFetch 抓不到时用浏览器的 `javascript_tool` 取 `document.body.innerText` 查关键行
5. **机械检查**:

```bash
f=<库>/<目录>/<材料>.md; echo "curly=$(grep -c '“\|”' $f) tilde=$(grep -c '~' $f) links=$(grep -cE '\]\((\.\./|/reports/[^)]*\.md)' $f) todo=$(grep -cE '知识库可以跟进|本文(刻意|不假装|不再复述)|零命中|pdfinfo' $f)"
```

   四项都应为 0;有「知识库可以跟进」一类段落就从正文删掉

小问题(一句过程措辞、一处页码区间写法)主会话直接改;事实层面的问题退回子 agent 或自己按原件改。

## 四、整批验证

工作区常有别的线未提交的改动,整库 `npm test` 的失败可能不是本批的。在只含「HEAD + 本批文件」的临时 worktree 里跑:

```bash
S=<scratchpad>; W=$S/wt && rm -rf $W && git worktree add -q --detach $W HEAD
for k in <公司/材料> ...; do cp reports/$k.md $W/reports/$k.md; cp -R public/reports/${k#*/} $W/public/reports/; done
ln -s $PWD/node_modules $W/node_modules && (cd $W && npm test > $S/t.log 2>&1); grep -E "Test Files|Tests |FAIL" $S/t.log
git worktree remove --force $W
```

- zsh 不按空格拆变量,篇目直接写在 for 列表里,不要放进一个字符串变量
- worktree 里没有未追踪的 PDF,`tests/original-pdf.test.ts` 那一项必然失败,属环境假失败;其余应全绿

然后在主工作区:

```bash
node scripts/svg-check.mjs public/reports/<slug> ...
npm run build        # 只写 .next-check
export LEETPREP_DIR=$PWD && source scripts/leet-server.sh && leet-restart   # 新配图必须重启才显示
```

页面检查在浏览器里对 `http://localhost:3000/reports` 执行下面这段,一次不超过 4 页(超过会超时):

```js
const pages=['<公司/材料>', ...]; const out=[];
for(const p of pages){ const f=document.createElement('iframe'); f.style.width='1200px'; f.style.height='800px'; f.src='/reports/'+p; document.body.appendChild(f);
  await new Promise(r=>{f.onload=r; setTimeout(r,8000)}); await new Promise(r=>setTimeout(r,3000));
  const d=f.contentDocument; const imgs=[...d.querySelectorAll('img')].filter(i=>i.src.includes('/reports/'));
  out.push({p, imgs:imgs.length, broken:imgs.filter(i=>i.naturalWidth===0).map(i=>i.src.split('/').pop()), merm:d.querySelectorAll('[class*=mermaid] svg').length,
    mermErr:d.querySelectorAll('.mermaid-error, [class*=mermaid][class*="red-"], [class*=mermaid] [class*="red-"]').length, katex:d.querySelectorAll('.katex-error').length,
    stars:((d.querySelector('article')||d.body).innerText.match(/\*\*/g)||[]).length}); f.remove(); }
JSON.stringify(out)
```

`broken`、`mermErr`、`katex`、`stars` 都应为空或 0;`merm` 为 0 时确认正文本来就没有 Mermaid 块。

## 五、执笔记录与提交

执笔记录是多线共享文件,常有别的线未提交的行。用脚本更新,它只暂存「HEAD + 本批这几行」,工作区同步改好且不会截断文件:

```bash
python3 .claude/skills/report-renovation/ledger.py --lib reports --date <今天> <公司/材料> ...   # 先加 --dry-run 看增量
git diff --cached --numstat reports/_执笔记录.md    # 行数应为 篇数 + 变动的汇总行数
```

然后只 `git add` 本批的正文与配图目录(配图目录里删掉的旧图也要一并暂存),提交并推送。**绝不 `git add -A`**。本线顺手改过的其他文件(知识库更正、兄弟篇加注)单独列进同一次提交并在提交信息里写明。

## 六、汇报与收尾

- 每批汇报:一张「篇目 | 旧稿里被纠正的主要问题」表,复核方式一句,需要维护者拍板的首发日或口径单列
- 子 agent 报的「知识库可能要跟进」汇总攒着,一个方向收完后集中补一次(按 docs/05 写,改前查题库联动)
- 一个方向收完:用执笔记录核一遍该方向已发布篇目是否全为「现行」或「标杆」
