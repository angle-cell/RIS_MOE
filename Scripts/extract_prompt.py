import json

def read_json(file_path, num):
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:                # 跳过空行
                prompts.append(line)
            if len(prompts) >= num:
                break
    # print(prompts[:1])
    # print(len(prompts))
    return prompts[:num]


def read_json_8k(file_path, num):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 顶层是字典，包含 "images" 等键
    
    prompts = []
    # 从顶层字典中获取 "images" 对应的列表（核心修复：定位到正确的数据列表）
    images_list = data.get("images", [])  # 假设句子数据在 "images" 键下
    
    # 遍历 images 列表中的每个元素（这些才是包含 "sentences" 的字典）
    for item in images_list:
        # 确保 item 是字典（防止异常数据）
        if not isinstance(item, dict):
            print(f"跳过 images 中的非字典元素: {item}（类型：{type(item)}）")
            continue
        
        # 提取当前 item 中的 sentences 列表
        sentences = item.get("sentences", [])
        for sentence in sentences:
            if isinstance(sentence, dict):
                raw_content = sentence.get("raw", "")
                if raw_content:
                    prompts.append(raw_content)
            else:
                print(f"sentences 中存在非字典元素: {sentence}（类型：{type(sentence)}）")
    
    # 去重并保留顺序
    prompts = list(dict.fromkeys(prompts))
    
    # 调试信息
    # print(f"前1条内容示例: {prompts[:1]}")
    # print(f"去重后总条数: {len(prompts)}")
    
    return prompts[:num]

# file_path = '/home/ygf/test/input/dataset_flickr8k.json'
# print(read_json_8k(file_path,10))