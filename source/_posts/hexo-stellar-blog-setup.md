---
title: 博客搭建记录：从零到一的 Hexo + Stellar 实践
date: 2025-07-24 10:30:00
tags: [hexo, stellar, 博客搭建, 静态网站, github-pages]
categories: [技术分享]
description: 详细记录使用 Hexo + Stellar 主题搭建个人博客的完整过程，包括环境配置、主题定制、部署优化等实践经验。
cover: /img/hexo-stellar-setup.jpg
toc: true
---

# 博客搭建记录：从零到一的 Hexo + Stellar 实践

作为一名技术人员，拥有一个个人博客来记录学习心得和分享技术经验是很有必要的。经过调研和实践，我最终选择了 Hexo + Stellar 主题的技术栈来搭建这个博客。

## 🎯 技术选型

### 为什么选择 Hexo？

在众多静态博客生成器中，我选择 Hexo 的原因：

1. **生态丰富**: 插件众多，扩展性强
2. **中文友好**: 中文文档完善，社区活跃
3. **性能优秀**: 生成速度快，支持增量生成
4. **主题丰富**: 有众多优秀的主题可选择

### 为什么选择 Stellar 主题？

Stellar 是我见过的最优雅的 Hexo 主题之一：

- **设计美观**: 现代化的设计风格，视觉效果出色
- **功能丰富**: 内置搜索、评论、统计等功能
- **移动友好**: 完美的响应式设计
- **配置灵活**: 高度可定制化
- **持续维护**: 作者 [@xaoxuu](https://github.com/xaoxuu) 持续更新

## 🛠️ 搭建过程

### 1. 环境准备

首先确保系统已安装 Node.js 和 Git：

```bash
# 检查 Node.js 版本
node --version

# 检查 npm 版本  
npm --version

# 检查 Git 版本
git --version
```

### 2. 安装 Hexo

```bash
# 全局安装 Hexo CLI
npm install -g hexo-cli

# 初始化博客项目
hexo init my-blog
cd my-blog

# 安装依赖
npm install
```

### 3. 安装 Stellar 主题

```bash
# 安装 Stellar 主题
npm install hexo-theme-stellar

# 或者使用 Git 克隆（推荐）
git clone https://github.com/xaoxuu/hexo-theme-stellar.git themes/stellar
```

### 4. 基础配置

修改根目录下的 `_config.yml`：

```yaml
# 站点信息
title: nash635
subtitle: 'For all time, always.'
description: '记录生活，分享技术，探索未知的可能性'
keywords: 技术博客,编程,开发,生活记录
author: nash635
language: zh-CN
timezone: 'Asia/Shanghai'

# URL 配置
url: https://nash635.github.io
permalink: :year/:month/:day/:title/

# 主题配置
theme: stellar
```

### 5. Stellar 主题配置

创建 `_config.stellar.yml` 文件进行主题配置：

```yaml
# 网站 Logo
logo:
  title: nash635
  subtitle: For all time, always.

# 导航菜单
menubar:
  columns: 4
  items:
    - name: 博客
      icon: solar:notebook-bold-duotone
      url: /
    - name: 项目
      icon: solar:code-bold-duotone
      url: /wiki/
    - name: 标签
      icon: solar:tag-bold-duotone
      url: /tags/
    - name: 关于
      icon: solar:user-bold-duotone
      url: /about/

# 搜索功能
search:
  service: local_search
  local_search:
    field: all
    path: /search.json
```

## 🎨 定制优化

### 1. 个性化配置

根据个人喜好调整主题配置：

```yaml
# 首页显示
home:
  title: 欢迎，旅行者
  subtitle: 在这里记录生活点滴，分享技术心得

# 文章页面
article:
  sidebar:
    position: right
    items:
      - widget: toc
      - widget: related_posts

# 页脚配置
footer:
  copyright: |
    本站由 @nash635 使用 [Stellar](https://github.com/xaoxuu/hexo-theme-stellar) 创建
```

### 2. 添加插件

安装常用插件增强功能：

```bash
# 搜索插件
npm install hexo-generator-search --save

# JSON 内容生成器
npm install hexo-generator-json-content --save

# 站点地图
npm install hexo-generator-sitemap --save

# RSS 订阅
npm install hexo-generator-feed --save
```

### 3. 创建页面

创建必要的页面：

```bash
# 关于页面
hexo new page about

# 标签页面
hexo new page tags

# 分类页面  
hexo new page categories

# 归档页面
hexo new page archives
```

## 🚀 部署优化

### 1. GitHub Pages 部署

配置 GitHub Actions 自动部署：

```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Node.js
      uses: actions/setup-node@v2
      with:
        node-version: '16'
        
    - name: Install dependencies
      run: npm install
      
    - name: Generate static files
      run: npm run build
      
    - name: Deploy
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./public
```

### 2. 性能优化

启用压缩和缓存：

```yaml
# _config.yml
# 压缩 HTML
minify:
  html:
    enable: true
    
# 启用 gzip 压缩
compress:
  html: true
  css: true
  js: true
```

## 📊 效果展示

经过配置优化后，博客具备了以下特性：

- ✅ **响应式设计**: 完美适配桌面端和移动端
- ✅ **搜索功能**: 支持本地搜索，快速查找内容
- ✅ **SEO 友好**: 良好的搜索引擎优化
- ✅ **加载迅速**: 静态文件，访问速度快
- ✅ **功能丰富**: 标签分类、归档、评论等功能完善

## 🎯 后续计划

博客搭建完成后，还有一些优化工作要做：

### 近期计划
- [ ] 配置评论系统（Giscus/Waline）
- [ ] 添加网站统计（Google Analytics）
- [ ] 优化 SEO 配置
- [ ] 添加 PWA 支持

### 长期计划
- [ ] 建立友链网络
- [ ] 开发自定义插件
- [ ] 添加更多实用工具页面
- [ ] 集成 API 服务

## 💡 经验总结

通过这次博客搭建，我总结了几点经验：

1. **选择合适的工具**: 根据需求选择技术栈，不要盲目追新
2. **注重用户体验**: 页面加载速度和移动端适配很重要
3. **持续优化**: 博客是一个长期项目，需要不断完善
4. **内容为王**: 再好的工具也要有优质的内容支撑

## 🔗 参考资源

- [Hexo 官方文档](https://hexo.io/docs/)
- [Stellar 主题文档](https://xaoxuu.com/wiki/stellar/)
- [GitHub Pages 文档](https://docs.github.com/pages)
- [Markdown 语法指南](https://markdown.com.cn/)

---

博客的搭建只是开始，接下来要做的是持续创作优质内容。如果你也想搭建类似的博客，希望这篇文章能对你有所帮助！

有任何问题欢迎在评论区交流～ 🤝
