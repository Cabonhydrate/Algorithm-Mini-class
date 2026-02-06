def get_max_val(matrix,x,y,kernel_size):
    max_val=matrix[y][x]
    for i in range(y,y+kernel_size):
        for j in range(x,x+kernel_size):
            max_val=max(max_val,matrix[i][j])
    
    return max_val


def max_pooling(matrix,kernel_size):
    if not isinstance(matrix, list) or len(matrix) == 0 or len(matrix[0]) == 0:
        print("Error: 输入矩阵为空或格式错误")
        return None
    
    width=len(matrix[0])
    height=len(matrix)

    
    if kernel_size <= 0 or not isinstance(kernel_size, int):
        print("Error: 池化核大小必须为正整数")
        return None
    if kernel_size > height or kernel_size > width:
        print(f"Error: 池化核大小({kernel_size})超过矩阵维度（高：{height}，宽：{width}）")
        return None
    
    output_width=width-kernel_size+1
    output_height=height-kernel_size+1
    res_matrix=[[0 for _ in range(output_width)] for _ in range(output_height)]
    
    for i in range(output_height):
        for j in range(output_width):
            res_matrix[i][j]=get_max_val(matrix,j,i,kernel_size )
    
    return res_matrix
