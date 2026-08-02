import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional

class SkillExecutor:
    """Skill 执行器：负责加载、解析和匹配 Skill"""

    def __init__(self, skills_dir: str = ".claude/skills"):
        """
        初始化 Skill 执行器

        Args:
            skills_dir: Skill 目录路径
        """
        self.skills_dir = Path(skills_dir)
        self.skills: Dict[str, Dict] = {}
        self.load_all_skills()

    def load_all_skills(self):
        """加载所有 Skill"""
        if not self.skills_dir.exists():
            print(f"Skill 目录不存在: {self.skills_dir}")
            return

        # 遍历 skills 目录下的所有子目录
        for skill_path in self.skills_dir.iterdir():
            if skill_path.is_dir():
                skill_file = skill_path / "SKILL.md"
                if skill_file.exists():
                    self.load_skill(skill_file)

    def load_skill(self, skill_file: Path):
        """
        加载单个 Skill

        Args:
            skill_file: SKILL.md 文件路径
        """
        with open(skill_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析 frontmatter 和 body
        frontmatter, body = self.parse_skill_content(content)

        if frontmatter and 'name' in frontmatter:
            skill_name = frontmatter['name']
            self.skills[skill_name] = {
                'name': skill_name,
                'description': frontmatter.get('description', ''),
                'body': body,
                'path': skill_file
            }
            print(f"✅ 加载 Skill: {skill_name}")

    def parse_skill_content(self, content: str) -> tuple:
        """
        解析 Skill 内容，分离 frontmatter 和 body

        Args:
            content: SKILL.md 文件内容

        Returns:
            (frontmatter_dict, body_string)
        """
        # 检查是否有 frontmatter（以 --- 开头和结尾）
        if not content.startswith('---'):
            return None, content

        # 分离 frontmatter 和 body
        parts = content.split('---', 2)
        if len(parts) < 3:
            return None, content

        # 解析 YAML frontmatter
        try:
            frontmatter = yaml.safe_load(parts[1])
        except yaml.YAMLError as e:
            print(f"❌ YAML 解析错误: {e}")
            return None, content

        body = parts[2].strip()
        return frontmatter, body

    def match_skill(self, user_request: str) -> Optional[Dict]:
        """
        根据用户请求匹配最合适的 Skill

        Args:
            user_request: 用户请求文本

        Returns:
            匹配到的 Skill 字典，如果没有匹配则返回 None
        """
        # 简化的匹配逻辑：基于关键词匹配
        keywords = user_request.lower().split()

        best_match = None
        best_score = 0

        for skill_name, skill_data in self.skills.items():
            description = skill_data['description'].lower()

            # 计算匹配分数
            score = sum(1 for keyword in keywords if keyword in description)

            if score > best_score:
                best_score = score
                best_match = skill_data

        if best_match and best_score > 0:
            print(f"🎯 匹配到 Skill: {best_match['name']} (匹配分数: {best_score})")
            return best_match

        print("❌ 未找到匹配的 Skill")
        return None