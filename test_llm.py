#!/usr/bin/env python3
"""
测试LLM解读功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_processor import LLMProcessor


def test_llm_processor():
    """测试LLM处理器"""
    try:
        # 初始化处理器
        processor = LLMProcessor()
        print("✅ LLM处理器初始化成功")
        
        # 检查API密钥
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("⚠️  警告: DEEPSEEK_API_KEY环境变量未设置")
            print("💡 请设置环境变量: export DEEPSEEK_API_KEY=your_api_key")
        
        # 测试论文数据
        test_papers = [
            {
                "title": "A new suite of Lund-tree observables to resolve jets",
                "abstract": "We introduce a class of collider observables, named Lund-Tree Shapes (LTS), defined from declustering trees originating from the Lund jet plane representation of the QCD radiation pattern in multi-jet scattering processes."
            },
            {
                "title": "WIMP Meets ALP: Coherent Freeze-Out of Dark Matter",
                "abstract": "We consider the cosmological history of a weakly interacting massive particle (WIMP) coupled to a light axion-like particle (ALP) via a quadratic coupling."
            }
        ]
        
        print(f"\n📝 测试 {len(test_papers)} 篇论文的LLM解读...")
        
        for i, paper in enumerate(test_papers, 1):
            print(f"\n--- 测试论文 {i} ---")
            print(f"标题: {paper['title']}")
            print(f"摘要: {paper['abstract'][:100]}...")
            
            # 生成解读
            interpretation = processor.generate_interpretation(
                paper["title"], paper["abstract"]
            )
            
            # 显示结果
            print(f"\n✅ 解读完成:")
            print(f"中文标题: {interpretation['chinese_translation']['title']}")
            print(f"中文摘要: {interpretation['chinese_translation']['abstract'][:100]}...")
            
            if interpretation['key_terms']:
                print(f"关键术语 ({len(interpretation['key_terms'])} 个):")
                for term in interpretation['key_terms']:
                    print(f"  - {term['term']} ({term['chinese']}): {term['explanation'][:80]}...")
            
            print("-" * 50)
        
        print("\n🎉 所有测试完成！LLM功能正常工作。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("\n💡 请检查:")
        print("1. DEEPSEEK_API_KEY环境变量是否设置")
        print("2. API密钥是否有效")
        print("3. 网络连接是否正常")
        return False
    
    return True


if __name__ == "__main__":
    print("🧪 开始测试LLM解读功能...")
    success = test_llm_processor()
    
    if success:
        print("\n✅ LLM解读功能测试通过！")
        print("\n📋 使用建议:")
        print("1. 运行 'python main.py --llm' 来使用LLM解读功能")
        print("2. 使用 '--max-papers' 参数控制处理数量")
        print("3. 查看 feed/ 目录下的输出文件")
    else:
        print("\n❌ LLM解读功能测试失败")
        sys.exit(1)