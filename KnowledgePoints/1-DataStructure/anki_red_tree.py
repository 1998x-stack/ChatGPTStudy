from pathlib import Path
import pandas as pd

# 构造红黑树插入操作的复习卡牌内容
flashcards = [
    {"Front": "红黑树中，红色节点的最大连续数量是？", "Back": "1（不能连续两个红）"},
    {"Front": "插入红黑树第一个节点，它的颜色是？", "Back": "黑色（根节点必须黑）"},
    {"Front": "哪种情况下插入后无需任何调整？", "Back": "父节点是黑色（Case 2）"},
    {"Front": "“父红 + 叔红”发生了，应该怎么做？", "Back": "上染红，父叔染黑（Case 3）"},
    {"Front": "“父红 + 叔黑 + 内插”，需要先做什么操作？", "Back": "先旋转成外插（Case 4）"},
    {"Front": "“父红 + 叔黑 + 外插”，需要做什么操作？", "Back": "爷爷旋转 + 父爷交换颜色（Case 5）"},
    {"Front": "插入过程中有没有可能递归往上修复？", "Back": "有，比如 Case 3 会往爷爷递归"},
    {"Front": "趣味题：插入 8 后出现红红相连，5 和 15 是红色节点，怎么办？", "Back": "颜色翻转：5 和 15 染黑，10 染红（Case 3）"},
    {"Front": "趣味题：插入 12，父 15 红，叔叔黑，当前是内插，怎么办？", "Back": "先旋转（右旋），变成 Case 5，再处理"},
]

# 创建 DataFrame 并导出为 TSV 文件（Anki 支持格式）
df = pd.DataFrame(flashcards)
anki_file_path = Path("red_black_tree_insert_flashcards.tsv")
df.to_csv(anki_file_path, sep="\t", index=False)

anki_file_path.name

