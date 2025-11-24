import urllib.request as libreq
import xml.etree.ElementTree as ET
import datetime
import time
import os

class DailyArxivFeed:
    def __init__(self, download_dir="daily_papers"):
        self.download_dir = download_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
    
    def get_date_range(self, days_back=0):
        """获取日期范围"""
        today = datetime.datetime.now()
        target_date = today - datetime.timedelta(days=days_back)
        date_str = target_date.strftime("%Y%m%d")
        return f"[{date_str} TO {date_str}]"
    
    def fetch_daily_papers(self, category=None, max_results=25, days_back=0):
        """获取每日新论文"""
        # 构建基础查询
        date_range = self.get_date_range(days_back)
        base_query = f"submittedDate:{date_range}"
        
        # 如果指定了分类，添加到查询中
        if category:
            search_query = f"cat:{category}+AND+{base_query}"
        else:
            search_query = base_query
        
        # 构建API URL - 使用urllib.parse来正确编码URL
        from urllib.parse import quote
        
        # 正确编码查询参数
        encoded_query = quote(search_query, safe='')
        api_url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query={encoded_query}&"
            f"sortBy=submittedDate&"
            f"sortOrder=descending&"
            f"max_results={max_results}"
        )
        
        print(f"🔍 查询每日arXiv论文")
        print(f"   日期范围: {date_range}")
        if category:
            print(f"   分类: {category}")
        print(f"   最大结果: {max_results}")
        print(f"   API URL: {api_url}")
        print("=" * 60)
        
        try:
            # 添加延迟避免请求过快
            time.sleep(1)
            
            with libreq.urlopen(api_url) as url:
                response = url.read()
            
            return self.parse_response(response, date_range, category)
            
        except Exception as e:
            print(f"❌ 获取失败: {e}")
            return None
    
    def parse_response(self, xml_content, date_range, category):
        """解析API响应"""
        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # 检查总结果数
            total_results_elem = root.find('.//{http://a9.com/-/spec/opensearch/1.1/}totalResults')
            total_results = int(total_results_elem.text) if total_results_elem is not None else 0
            
            papers = []
            for entry in root.findall('atom:entry', ns):
                paper_info = {
                    'title': entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else "Unknown Title",
                    'id': entry.find('atom:id', ns).text,
                    'summary': entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else "",
                    'published': entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else "",
                    'updated': entry.find('atom:updated', ns).text if entry.find('atom:updated', ns) is not None else "",
                    'pdf_url': None,
                    'authors': []
                }
                
                # 提取作者
                for author in entry.findall('atom:author/atom:name', ns):
                    paper_info['authors'].append(author.text)
                
                # 提取分类
                paper_info['categories'] = []
                for cat in entry.findall('atom:category', ns):
                    paper_info['categories'].append(cat.get('term'))
                
                # 查找PDF链接
                for link in entry.findall('atom:link', ns):
                    if link.get('title') == 'pdf':
                        paper_info['pdf_url'] = link.get('href')
                        break
                
                papers.append(paper_info)
            
            return {
                'total_results': total_results,
                'papers': papers,
                'date_range': date_range,
                'category': category
            }
            
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            return None
    
    def display_results(self, results):
        """显示结果"""
        if not results:
            print("❌ 无结果")
            return
        
        print(f"\n📊 搜索结果:")
        print(f"   总共找到: {results['total_results']} 篇论文")
        print(f"   返回了: {len(results['papers'])} 篇论文")
        print(f"   日期范围: {results['date_range']}")
        if results['category']:
            print(f"   分类: {results['category']}")
        print("-" * 60)
        
        for i, paper in enumerate(results['papers'], 1):
            print(f"\n[{i}] {paper['title']}")
            print(f"   作者: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
            print(f"   发布时间: {paper['published']}")
            print(f"   分类: {', '.join(paper['categories'][:2])}{'...' if len(paper['categories']) > 2 else ''}")
            if paper['pdf_url']:
                print(f"   PDF: {paper['pdf_url']}")
            
            # 显示摘要前100个字符
            if paper['summary']:
                summary_preview = paper['summary'][:100] + "..." if len(paper['summary']) > 100 else paper['summary']
                print(f"   摘要: {summary_preview}")
    
    def download_selected_papers(self, results, indices=None):
        """下载选定的论文"""
        if not results or not results['papers']:
            print("❌ 无论文可下载")
            return
        
        if indices is None:
            indices = range(len(results['papers']))
        
        downloaded_count = 0
        for idx in indices:
            if 0 <= idx < len(results['papers']):
                paper = results['papers'][idx]
                if paper['pdf_url']:
                    if self.download_paper(paper):
                        downloaded_count += 1
        
        print(f"\n✅ 下载完成: {downloaded_count} 篇论文")
    
    def download_paper(self, paper_info):
        """下载单个论文PDF"""
        if not paper_info['pdf_url']:
            print(f"✗ 跳过 '{paper_info['title']}' - 无PDF链接")
            return False
        
        try:
            # 创建更好的文件名
            year = paper_info['published'][:4] if paper_info['published'] else "unknown"
            title_words = paper_info['title'].split()[:4]
            clean_title = "_".join(title_words).lower()
            filename = f"{year}_{clean_title}.pdf"
            
            # 清理文件名中的非法字符
            invalid_chars = '<>:"/\\|?*'
            for char in invalid_chars:
                filename = filename.replace(char, '_')
            
            filepath = os.path.join(self.download_dir, filename)
            
            print(f"📥 正在下载: {paper_info['title']}")
            libreq.urlretrieve(paper_info['pdf_url'], filepath)
            print(f"   ✅ 保存为: {filename}")
            return True
            
        except Exception as e:
            print(f"   ❌ 下载失败: {e}")
            return False

def main():
    """主函数 - 演示每日arXiv订阅功能"""
    feed = DailyArxivFeed()
    
    print("🚀 arXiv每日论文订阅系统")
    print("=" * 50)
    
    # 示例：获取今天的新论文
    print("\n1. 获取今天的新论文:")
    today_results = feed.fetch_daily_papers(max_results=10, days_back=0)
    feed.display_results(today_results)
    
    # 示例：获取特定分类的新论文
    print("\n" + "="*60)
    print("2. 获取计算机科学-人工智能分类的新论文:")
    cs_ai_results = feed.fetch_daily_papers(category="cs.AI", max_results=5, days_back=0)
    feed.display_results(cs_ai_results)
    
    # 示例：获取昨天的论文
    print("\n" + "="*60)
    print("3. 获取昨天的论文:")
    yesterday_results = feed.fetch_daily_papers(max_results=5, days_back=1)
    feed.display_results(yesterday_results)
    
    # 可选：下载一些论文
    print("\n" + "="*60)
    print("4. 下载前2篇论文:")
    if today_results and today_results['papers']:
        feed.download_selected_papers(today_results, indices=[0, 1])

if __name__ == "__main__":
    main()