import feedparser
import requests
import os
import json
from datetime import datetime
from typing import List, Dict, Any
from llm_processor import BatchLLMProcessor, save_interpretation_cache, load_interpretation_cache


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
            
            # 添加LLM解读内容
            if "llm_interpretation" in paper:
                interpretation = paper["llm_interpretation"]
                
                # 中文翻译
                markdown_content += "### 中文翻译\n\n"
                markdown_content += f"**标题**: {interpretation['chinese_translation']['title']}\n\n"
                markdown_content += f"**摘要**: {interpretation['chinese_translation']['abstract']}\n\n"
                
                # 关键术语解释
                if interpretation['key_terms']:
                    markdown_content += "### 关键术语解释\n\n"
                    for term in interpretation['key_terms']:
                        markdown_content += f"**{term['term']}** ({term['chinese']})\n\n"
                        markdown_content += f"{term['explanation']}\n\n"
            
            markdown_content += "---\n\n"
    else:
        markdown_content += "今天没有找到新的论文\n"
    
    # 写入文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"已保存到: {file_path}")


def main(enable_llm: bool = False, max_papers_per_category: int = 5):
    """
    主函数
    
    Args:
        enable_llm: 是否启用LLM解读
        max_papers_per_category: 每个类别处理的最大论文数量
    """
    from datetime import datetime
    
    # 定义要抓取的类别
    categories = ["astro-ph","cond-mat","gr-qc","hep-ex","hep-lat","hep-ph", "hep-th","math-ph","nlin","nucl-ex","nucl-th","physics","quant-ph","math","q-bio","q-fin","stat","eess", "econ"]
    today = datetime.now().strftime('%Y-%m-%d')
    
    print(f"开始抓取arXiv今日({today})发布的论文...")
    print(f"目标类别: {', '.join(categories)}")
    print(f"LLM解读: {'启用' if enable_llm else '禁用'}")
    if enable_llm:
        print(f"每个类别最大处理论文数: {max_papers_per_category}")
    
    all_results = {}
    
    # 初始化LLM处理器
    llm_processor = None
    if enable_llm:
        try:
            llm_processor = BatchLLMProcessor()
            print("LLM处理器初始化成功")
        except Exception as e:
            print(f"LLM处理器初始化失败: {e}")
            print("将继续处理但不使用LLM解读")
            enable_llm = False
    
    for category in categories:
        print(f"\n正在处理 {category} 类别...")
        
        try:
            # 获取今天发布的论文
            today_papers = fetch_todays_arxiv_papers(category)
            
            print(f"  {category}: 找到 {len(today_papers)} 篇论文")
            
            # 如果启用LLM且论文数量超过限制，则限制处理数量
            if enable_llm and len(today_papers) > max_papers_per_category:
                print(f"  {category}: 限制处理前 {max_papers_per_category} 篇论文")
                today_papers = today_papers[:max_papers_per_category]
            
            # 使用LLM处理论文
            if enable_llm and llm_processor and today_papers:
                print(f"  {category}: 开始LLM解读...")
                today_papers = llm_processor.process_papers(today_papers)
                print(f"  {category}: LLM解读完成")
            
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


def main_with_llm():
    """启用LLM解读的主函数"""
    return main(enable_llm=True, max_papers_per_category=3)


def main_without_llm():
    """禁用LLM解读的主函数"""
    return main(enable_llm=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="arXiv论文抓取工具")
    parser.add_argument("--llm", action="store_true", help="启用LLM解读")
    parser.add_argument("--max-papers", type=int, default=3, help="每个类别处理的最大论文数量（仅当启用LLM时有效）")
    
    args = parser.parse_args()
    
    if args.llm:
        print("启用LLM解读模式")
        main(enable_llm=True, max_papers_per_category=args.max_papers)
    else:
        print("使用普通模式（无LLM解读）")
        main(enable_llm=False)
