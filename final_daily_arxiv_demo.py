import urllib.request as libreq
import xml.etree.ElementTree as ET
import datetime
from urllib.parse import quote
import os

def demonstrate_daily_arxiv_feed():
    """演示完整的每日arXiv订阅功能"""
    
    print("🎯 arXiv每日论文订阅系统 - 完整演示")
    print("=" * 60)
    
    # 测试一个已知有论文的日期
    test_date = "20241120"  # 几天前，确保有数据
    date_range = f"[{test_date} TO {test_date}]"
    
    print(f"\n📅 获取 {test_date} 的新论文:")
    print("-" * 40)
    
    # 1. 获取所有分类的新论文
    search_query = f"submittedDate:{date_range}"
    encoded_query = quote(search_query, safe='')
    
    api_url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query={encoded_query}&"
        f"sortBy=submittedDate&"
        f"sortOrder=descending&"
        f"max_results=10"
    )
    
    print(f"🔗 API URL: {api_url}")
    
    try:
        with libreq.urlopen(api_url) as url:
            response = url.read()
        
        root = ET.fromstring(response)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        # 检查总结果数
        total_results_elem = root.find('.//{http://a9.com/-/spec/opensearch/1.1/}totalResults')
        total_results = int(total_results_elem.text) if total_results_elem is not None else 0
        
        papers = root.findall('atom:entry', ns)
        
        print(f"📊 统计: 总共 {total_results} 篇论文，返回 {len(papers)} 篇")
        print("\n📚 论文列表:")
        
        for i, paper in enumerate(papers, 1):
            title = paper.find('atom:title', ns).text.strip() if paper.find('atom:title', ns) is not None else "Unknown"
            published = paper.find('atom:published', ns).text if paper.find('atom:published', ns) is not None else "Unknown"
            
            # 提取分类
            categories = []
            for cat in paper.findall('atom:category', ns):
                categories.append(cat.get('term'))
            
            # 提取作者
            authors = []
            for author in paper.findall('atom:author/atom:name', ns):
                authors.append(author.text)
            
            # 查找PDF链接
            pdf_url = None
            for link in paper.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break
            
            print(f"\n[{i}] {title}")
            print(f"   作者: {', '.join(authors[:2])}{'...' if len(authors) > 2 else ''}")
            print(f"   分类: {', '.join(categories[:2])}{'...' if len(categories) > 2 else ''}")
            print(f"   发布时间: {published}")
            if pdf_url:
                print(f"   📄 PDF可用: {pdf_url}")
        
        # 2. 演示特定分类的搜索
        print(f"\n" + "="*60)
        print("🔬 按分类搜索 (计算机科学-人工智能):")
        print("-" * 40)
        
        category_search_query = f"cat:cs.AI+AND+submittedDate:{date_range}"
        encoded_category_query = quote(category_search_query, safe='')
        
        category_api_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={encoded_category_query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"max_results=5"
        )
        
        print(f"🔗 API URL: {category_api_url}")
        
        with libreq.urlopen(category_api_url) as url:
            category_response = url.read()
        
        category_root = ET.fromstring(category_response)
        category_papers = category_root.findall('atom:entry', ns)
        
        category_total_elem = category_root.find('.//{http://a9.com/-/spec/opensearch/1.1/}totalResults')
        category_total = int(category_total_elem.text) if category_total_elem is not None else 0
        
        print(f"📊 统计: 总共 {category_total} 篇AI论文，返回 {len(category_papers)} 篇")
        
        for i, paper in enumerate(category_papers, 1):
            title = paper.find('atom:title', ns).text.strip() if paper.find('atom:title', ns) is not None else "Unknown"
            print(f"   {i}. {title[:70]}...")
        
        # 3. 演示论文下载功能
        print(f"\n" + "="*60)
        print("💾 论文下载演示:")
        print("-" * 40)
        
        download_dir = "demo_papers"
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        
        # 下载第一篇论文
        if papers:
            first_paper = papers[0]
            pdf_url = None
            for link in first_paper.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break
            
            if pdf_url:
                title = first_paper.find('atom:title', ns).text.strip()
                print(f"📥 正在下载: {title[:50]}...")
                
                # 创建文件名
                year = first_paper.find('atom:published', ns).text[:4] if first_paper.find('atom:published', ns) is not None else "unknown"
                title_words = title.split()[:3]
                clean_title = "_".join(title_words).lower()
                filename = f"{year}_{clean_title}.pdf"
                
                # 清理文件名
                invalid_chars = '<>:"/\\|?*'
                for char in invalid_chars:
                    filename = filename.replace(char, '_')
                
                filepath = os.path.join(download_dir, filename)
                
                try:
                    libreq.urlretrieve(pdf_url, filepath)
                    print(f"✅ 下载完成: {filename}")
                    print(f"📁 保存位置: {download_dir}/")
                except Exception as e:
                    print(f"❌ 下载失败: {e}")
            else:
                print("❌ 第一篇论文无PDF链接")
        else:
            print("❌ 无论文可下载")
        
        print(f"\n" + "="*60)
        print("🎉 演示完成!")
        print("\n📋 总结:")
        print("   ✅ 可以获取每日新论文")
        print("   ✅ 支持按分类筛选")
        print("   ✅ 可以下载PDF论文")
        print("   ✅ 支持日期范围搜索")
        print("   ✅ 可以按提交时间排序")
        print(f"\n💡 使用建议:")
        print("   1. 每天运行一次获取最新论文")
        print("   2. 设置感兴趣的分类过滤")
        print("   3. 自动下载感兴趣的论文")
        print("   4. 可以集成到邮件通知系统")
        
    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    demonstrate_daily_arxiv_feed()