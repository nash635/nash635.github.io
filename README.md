# nash635's Blog

[![Deploy to GitHub Pages](https://github.com/nash635/nash635.github.io/actions/workflows/deploy.yml/badge.svg)](https://github.com/nash635/nash635.github.io/actions/workflows/deploy.yml)

> For all time, always.

这是我的个人博客，基于 Hexo + Stellar 主题构建，用于记录技术学习心得和生活感悟。

## 🌐 访问地址

- **主站**: [nash635.github.io](https://nash635.github.io)

## 🛠️ 技术栈

- **生成器**: [Hexo 7.3.0](https://hexo.io/)
- **主题**: [Stellar 1.33.1](https://github.com/xaoxuu/hexo-theme-stellar)
- **部署**: GitHub Pages + GitHub Actions
- **搜索**: 本地搜索
- **评论**: 待配置

## 📂 目录结构

```
├── source/              # 源文件目录
│   ├── _posts/         # 博客文章
│   ├── about/          # 关于页面
│   ├── wiki/           # 项目页面
│   ├── notes/          # 探索页面
│   ├── friends/        # 友链页面
│   ├── topic/          # 专栏页面
│   └── img/            # 图片资源
├── themes/             # 主题文件
├── _config.yml         # Hexo 配置
├── _config.stellar.yml # Stellar 主题配置
└── package.json        # 项目依赖
```

## 🚀 本地开发

```bash
# 克隆项目
git clone https://github.com/nash635/nash635.github.io.git
cd nash635.github.io

# 安装依赖
npm install

# 启动开发服务器
npm run dev

# 构建静态文件
npm run build

# 创建新文章
npm run new "文章标题"
```

## 📝 内容板块

- **[博客](/)**: 技术文章和生活感悟
- **[项目](/wiki/)**: 开源项目和个人作品
- **[探索](/notes/)**: 学习笔记和思考记录
- **[社交](/friends/)**: 友情链接和交流平台
- **[专栏](/topic/)**: 系列文章和深度内容
- **[标签](/tags/)**: 按标签浏览文章
- **[分类](/categories/)**: 按分类查看内容
- **[归档](/archives/)**: 时间线浏览历史文章
- **[关于](/about/)**: 个人介绍和联系方式

## 🎯 特色功能

- ✅ 响应式设计，完美适配各种设备
- ✅ 本地搜索，快速查找内容
- ✅ 文章目录，便于阅读导航
- ✅ 代码高亮，支持多种编程语言
- ✅ 数学公式，支持 LaTeX 语法
- ✅ 图片懒加载，优化加载性能
- ✅ SEO 优化，提升搜索引擎排名

## 📞 联系方式

- **GitHub**: [@nash635](https://github.com/nash635)
- **Email**: nash635@example.com
- **Blog**: [nash635.github.io](https://nash635.github.io)


## 📦 发布流程

本博客使用 **gh-pages 分支** 部署到 GitHub Pages。

### 快速发布

```bash
# 1. 写文章并提交到 master
git add source/_posts/新文章.md
git commit -m "Add: 文章标题"
git push

# 2. 构建并部署
npm run build
cp -r public /tmp/hp
git checkout gh-pages
rm -rf _config* package* scaffolds source themes node_modules db.json .github README.md .gitignore public
cp -r /tmp/hp/* . && rm -rf /tmp/hp
git add -A && git commit -m "Deploy: 文章标题" && git push
git checkout master
```

### GitHub Pages 设置

确保 Settings → Pages → Branch 设置为 **gh-pages/(root)**

## 🙏 致谢

感谢以下开源项目：

- [Hexo](https://hexo.io/) - 快速、简洁且高效的博客框架
- [Stellar](https://github.com/xaoxuu/hexo-theme-stellar) - 优雅的 Hexo 主题
- [GitHub Pages](https://pages.github.com/) - 免费的静态网站托管

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

---

**欢迎 Star ⭐ 和 Fork 🍴 这个项目！**
