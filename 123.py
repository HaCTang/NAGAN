from rdkit import Chem

def sdf_to_smiles(sdf_file, output_file=None):
    """
    将 SDF 文件中的分子转换为 SMILES 表示。
    
    参数:
    sdf_file (str): SDF 文件路径。
    output_file (str, optional): 如果提供，则将 SMILES 结果保存到该文件中。
    
    返回:
    list: 包含所有分子的 SMILES 结构的列表。
    """
    supplier = Chem.SDMolSupplier(sdf_file)
    smiles_list = []

    for mol in supplier:
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            smiles_list.append(smiles)
    
    # 如果提供了输出文件，则将 SMILES 写入文件
    if output_file:
        with open(output_file, 'w') as f:
            for smiles in smiles_list:
                f.write(smiles + '\n')
    
    return smiles_list

# 使用示例：
sdf_file = 'your_file.sdf'
output_file = 'output_smiles.txt'
smiles_list = sdf_to_smiles(sdf_file, output_file)

# 如果不需要保存到文件，直接调用：
# smiles_list = sdf_to_smiles(sdf_file)



from rdkit import Chem
from rdkit.Chem import Draw

def sdf_to_images(sdf_file, output_dir, image_size=(300, 300)):
    """
    将 SDF 文件中的分子转换为 2D 图片并保存。
    
    参数:
    sdf_file (str): SDF 文件路径。
    output_dir (str): 保存图片的文件夹路径。
    image_size (tuple, optional): 图片大小，默认是 (300, 300)。
    
    返回:
    None
    """
    supplier = Chem.SDMolSupplier(sdf_file)
    
    for idx, mol in enumerate(supplier):
        if mol is not None:
            # 生成2D坐标
            Chem.rdDepictor.Compute2DCoords(mol)
            
            # 生成分子的2D图片
            img = Draw.MolToImage(mol, size=image_size)
            
            # 图片保存路径
            img_path = f"{output_dir}/mol_{idx + 1}.png"
            
            # 保存图片
            img.save(img_path)
            print(f"Saved image {idx + 1} to {img_path}")

# 使用示例：
sdf_file = 'your_file.sdf'
output_dir = 'output_images'
sdf_to_images(sdf_file, output_dir)



from rdkit import Chem

def is_valid_smiles(smiles):
    """
    判断给定的 SMILES 是否合理。
    
    参数:
    smiles (str): 要验证的 SMILES 字符串。
    
    返回:
    bool: 如果 SMILES 合法，返回 True；否则返回 False。
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# 使用示例：
smiles = "CCO"  # 合法的 SMILES
invalid_smiles = "C1CC1C"  # 不合法的 SMILES（环闭合错误）

print(is_valid_smiles(smiles))        # 输出: True
print(is_valid_smiles(invalid_smiles))  # 输出: False


# 安装osra, osra -s input_image.png
import subprocess

def image_to_smiles(image_file):
    """
    使用 OSRA 将分子图片转换为 SMILES。
    
    参数:
    image_file (str): 分子图片的文件路径。
    
    返回:
    str: 解析得到的 SMILES 字符串。
    """
    try:
        result = subprocess.run(['osra', '-s', image_file], capture_output=True, text=True)
        smiles = result.stdout.strip()  # 提取 SMILES
        return smiles
    except Exception as e:
        print(f"Error: {e}")
        return None

# 使用示例：
image_file = 'molecule.png'
smiles = image_to_smiles(image_file)
print(f"SMILES: {smiles}")