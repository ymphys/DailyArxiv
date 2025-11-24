import feedparser
import requests
from datetime import datetime
from typing import List, Dict, Any


def fetch_todays_arxiv_papers(category: str = "hep-ph") -> List[Dict[str, Any]]:
    """
    获取今天在arXiv上发布的论文（模拟访问new页面）
    
    Args:
        category: arXiv类别，默认为hep-ph
    
    Returns:
        论文信息列表，每个论文包含title, abstract, authors, pdf_url, published
    """
    from datetime import datetime
    
    # 直接访问arXiv的new页面
    new_url = f"https://arxiv.org/list/{category}/new"
    
    try:
        response = requests.get(new_url)
        response.raise_for_status()
        
        # 使用BeautifulSoup解析HTML
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        papers = []
        
        # 查找论文列表
        dt_list = soup.find_all('dt')
        dd_list = soup.find_all('dd')
        
        for dt, dd in zip(dt_list, dd_list):
            try:
                # 提取论文ID和标题
                paper_id = dt.find('a', title='Abstract')['href'].split('/')[-1]
                
                # 提取标题
                title_div = dd.find('div', class_='list-title mathjax')
                if title_div:
                    title = title_div.text.replace('Title:', '').strip()
                else:
                    continue
                
                # 提取作者
                authors_div = dd.find('div', class_='list-authors')
                authors = []
                if authors_div:
                    author_links = authors_div.find_all('a')
                    authors = [author.text.strip() for author in author_links]
                
                # 提取摘要
                abstract_p = dd.find('p', class_='mathjax')
                abstract = abstract_p.text.strip() if abstract_p else ""
                
                # 构建论文信息
                paper = {
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
                    "published": datetime.now().strftime('%Y-%m-%d')  # 假设今天发布的
                }
                
                papers.append(paper)
                
            except Exception as e:
                print(f"解析论文时出错: {e}")
                continue
        
        return papers
        
    except Exception as e:
        print(f"获取今天发布的论文时出错: {e}")
        return []


def fetch_arxiv_papers(category: str = "hep-ph", max_results: int = 50) -> List[Dict[str, Any]]:
    """
    从arXiv API获取指定类别的论文（备用方法）
    
    Args:
        category: arXiv类别，默认为hep-ph
        max_results: 最大返回结果数，默认为50
    
    Returns:
        论文信息列表，每个论文包含title, abstract, authors, pdf_url, published
    """
    # 构建API URL - 获取最新发布的论文（按更新时间排序）
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"cat:{category}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "lastUpdatedDate",
        "sortOrder": "descending"
    }
    
    # 发送请求
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    # 解析Atom feed
    feed = feedparser.parse(response.content)
    
    papers = []
    for entry in feed.entries:
        # 提取论文信息
        paper = {
            "title": entry.title.replace("\n", " ").strip(),
            "abstract": entry.summary.replace("\n", " ").strip(),
            "authors": [author.name for author in entry.authors],
            "pdf_url": None,
            "published": None
        }
        
        # 查找PDF链接
        for link in entry.links:
            if link.rel == "alternate" and link.type == "text/html":
                # 这是论文页面链接，我们可以从中提取PDF链接
                pdf_url = link.href.replace("/abs/", "/pdf/") + ".pdf"
                paper["pdf_url"] = pdf_url
                break
            elif link.type == "application/pdf":
                paper["pdf_url"] = link.href
                break
        
        # 解析发布日期
        if hasattr(entry, 'published'):
            try:
                published_date = datetime.strptime(entry.published, '%Y-%m-%dT%H:%M:%SZ')
                paper["published"] = published_date.strftime('%Y-%m-%d')
            except ValueError:
                paper["published"] = entry.published
        
        papers.append(paper)
    
    return papers


def save_papers_to_markdown(papers: List[Dict[str, Any]], category: str, date: str):
    """
    将论文列表保存为Markdown文件
    
    Args:
        papers: 论文列表
        category: arXiv类别
        date: 日期
    """
    import os
    
    # 创建目录
    dir_path = f"./feed/{category}"
    os.makedirs(dir_path, exist_ok=True)
    
    # 文件路径
    file_path = f"{dir_path}/{date}-arxiv.md"
    
    # 生成Markdown内容
    markdown_content = f"# arXiv {category} 今日论文 ({date})\n\n"
    
    if papers:
        markdown_content += f"共找到 {len(papers)} 篇论文\n\n"
        
        for i, paper in enumerate(papers, 1):
            markdown_content += f"## {i}. {paper['title']}\n\n"
            markdown_content += f"**作者**: {', '.join(paper['authors'])}\n\n"
            markdown_content += f"**PDF链接**: [{paper['pdf_url']}]({paper['pdf_url']})\n\n"
            markdown_content += f"**摘要**: {paper['abstract']}\n\n"
            markdown_content += "---\n\n"
    else:
        markdown_content += "今天没有找到新的论文\n"
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"已保存到: {file_path}")


def main():
    """主函数"""
    from datetime import datetime
    
    # 定义要抓取的类别
    categories = ["astro-ph","cond-mat","gr-qc","hep-ex","hep-lat","hep-ph", "hep-th","math-ph","nlin","nucl-ex","nucl-th","physics","quant-ph","math","q-bio","q-fin","stat","eess", "econ"]
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"开始抓取arXiv今日({today})发布的论文...")
    print(f"目标类别: {', '.join(categories)}")
    
    all_results = {}
    
    for category in categories:
        print(f"\n正在处理 {category} 类别...")
        
        try:
            # 获取今天发布的论文
            today_papers = fetch_todays_arxiv_papers(category)
            
            print(f"  {category}: 找到 {len(today_papers)} 篇论文")
            
            # 保存为Markdown文件
            save_papers_to_markdown(today_papers, category, today)
            
            all_results[category] = today_papers
            
        except requests.RequestException as e:
            print(f"  {category}: 网络请求错误 - {e}")
            all_results[category] = []
        except Exception as e:
            print(f"  {category}: 发生错误 - {e}")
            all_results[category] = []
    
    print(f"\n完成！所有类别的论文已保存到 ./feed/ 目录")
    
    # 返回所有结果供其他程序使用
    return all_results


if __name__ == "__main__":
    main()
