# Assignment
Solutions for NumPy + Pandas assignment
import numpy as np
import pandas as pd

# Q1 - Create a NumPy array 'arr' of integers from 0 to 5 and print its data type.
arr = np.arange(6)
print("Q1:", arr.dtype)

# Q2 - Given a NumPy array 'arr', check if its data type is float64.
print("Q2:", arr.dtype == np.float64)

# Q3 - Create a NumPy array 'arr' with a data type of complex128 containing three complex numbers.
arr = np.array([1+2j, 3+4j, 5+6j], dtype=np.complex128)
print("Q3:", arr)

# Q4 - Convert an existing NumPy array 'arr' of integers to float32 data type.
arr = np.arange(6)
print("Q4:", arr.astype(np.float32))

# Q5 - Given a NumPy array 'arr' with float64 data type, convert it to float32 to reduce decimal precision.
arr = np.array([1.2, 2.4, 3.6], dtype=np.float64)
print("Q5:", arr.astype(np.float32))

# Q6 - Write a function array_attributes that returns shape, size, and dtype.
def array_attributes(arr): return arr.shape, arr.size, arr.dtype
print("Q6:", array_attributes(np.array([[1,2],[3,4]])))

# Q7 - Function to return array dimension.
def array_dimension(arr): return arr.ndim
print("Q7:", array_dimension(np.array([[1,2],[3,4]])))

# Q8 - Function to return item size and total size in bytes.
def item_size_info(arr): return arr.itemsize, arr.nbytes
print("Q8:", item_size_info(np.array([[1,2],[3,4]])))

# Q9 - Function to return strides of array.
def array_strides(arr): return arr.strides
print("Q9:", array_strides(np.array([1.5, 2.6, 3.7])))

# Q10 - Function to return shape and strides.
def shape_stride_relationship(arr): return arr.shape, arr.strides
print("Q10:", shape_stride_relationship(np.array([[1,2,3],[4,5,6]])))

# Q11 - Function to create a zeros array of n elements.
def create_zeros_array(n): return np.zeros(n)
print("Q11:", create_zeros_array(5))

# Q12 - Function to create 2D array of ones.
def create_ones_matrix(rows, cols): return np.ones((rows, cols))
print("Q12:", create_ones_matrix(2,3))

# Q13 - Function to generate array from start to stop with step.
def generate_range_array(start, stop, step): return np.arange(start, stop, step)
print("Q13:", generate_range_array(0,10,2))

# Q14 - Generate array with num equally spaced values from start to stop.
def generate_linear_space(start, stop, num): return np.linspace(start, stop, num)
print("Q14:", generate_linear_space(1.0, 5.0, 5))

# Q15 - Create identity matrix of size n x n.
def create_identity_matrix(n): return np.eye(n)
print("Q15:", create_identity_matrix(3))

# Q16 - Convert Python list to NumPy array.
def list_to_array(lst): return np.array(lst)
print("Q16:", list_to_array([1,2,3]))

# Q17 - Demonstrate use of numpy.view.
arr = np.array([1,2,3])
print("Q17:", arr.view())

# Q18 - Concatenate two arrays along given axis.
def concat_arrays(a1, a2, axis=0): return np.concatenate((a1, a2), axis=axis)
print("Q18:", concat_arrays(np.array([1,2]), np.array([3,4])))

# Q19 - Concatenate two arrays of different shapes horizontally.
a = np.array([[1],[2]])
b = np.array([[3,4]])
print("Q19:", np.concatenate((a, b.T), axis=1))

# Q20 - Vertically stack NumPy arrays.
def vertical_stack(arrays): return np.vstack(arrays)
print("Q20:", vertical_stack([np.array([1,2]), np.array([3,4])]))

# Q21 - Create array of integers in range with step (inclusive).
def range_step(start, stop, step): return np.arange(start, stop+1, step)
print("Q21:", range_step(1, 10, 2))

# Q22 - Generate 10 equally spaced values from 0 to 1.
def equal_0_1(): return np.linspace(0,1,10)
print("Q22:", equal_0_1())

# Q23 - Create array of 5 log spaced values from 1 to 1000.
def log_spaced(): return np.logspace(0, 3, 5)
print("Q23:", log_spaced())

# Q24 - Create DataFrame of random ints (5 rows, 3 cols).
df = pd.DataFrame(np.random.randint(1,100,size=(5,3)))
print("Q24:\n", df)

# Q25 - Replace negative values with 0 in a specific column.
def replace_negatives(df, col): df[col] = df[col].apply(lambda x: 0 if x<0 else x); return df
print("Q25:\n", replace_negatives(pd.DataFrame({'A': [1, -2, 3]}), 'A'))

# Q26 - Access the 3rd element of an array.
arr = np.array([10, 20, 30, 40, 50])
print("Q26:", arr[2])

# Q27 - Retrieve element at (1, 2) in 2D array.
arr_2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Q27:", arr_2d[1,2])

# Q28 - Extract elements > 5 using boolean indexing.
arr = np.array([3, 8, 2, 10, 5, 7])
print("Q28:", arr[arr > 5])

# Q29 - Slice array from index 2 to 5 (inclusive).
arr = np.array([1,2,3,4,5,6,7,8,9])
print("Q29:", arr[2:6])

# Q30 - Extract subarray [[2, 3], [5, 6]] from arr_2d.
arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Q30:", arr[0:2, 1:3])

# Q31 - Extract elements using flat indices.
def extract_by_indices(arr, indices): return arr.flat[indices]
print("Q31:", extract_by_indices(np.array([[1,2],[3,4]]), [0,3]))

# Q32 - Filter elements > threshold using boolean indexing.
def filter_threshold(arr, th): return arr[arr > th]
print("Q32:", filter_threshold(np.array([1,2,3,4]), 2))

# Q33 - Extract specific elements using multi-dim indices.
arr = np.arange(27).reshape(3,3,3)
print("Q33:", arr[[0,1],[1,2],[2,0]])

# Q34 - Return elements satisfying two boolean conditions.
def two_conditions(arr): return arr[(arr>2) & (arr<7)]
print("Q34:", two_conditions(np.array([1,3,5,7])))

# Q35 - Extract using separate row and col indices.
def extract_rc(arr, r, c): return arr[r, c]
print("Q35:", extract_rc(np.array([[1,2],[3,4]]), [0,1], [1,0]))

# Q36 - Add scalar to each element using broadcasting.
arr = np.ones((3,3))
print("Q36:", arr + 5)

# Q37 - Multiply each row using broadcasting.
arr1 = np.array([[1,2,3]])
arr2 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("Q37:", arr2 * arr1.T)

# Q38 - Add 1D array to each row of 2D array.
arr1 = np.array([[1,2,3,4]])
arr2 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
print("Q38:", arr2 + arr1.T)

# Q39 - Add 3x1 and 1x3 arrays using broadcasting.
arr1 = np.array([[1],[2],[3]])
arr2 = np.array([[4,5,6]])
print("Q39:", arr1 + arr2)

# Q40 - Handle shape incompatibility for broadcasting.
try:
    arr1 = np.ones((2,3))
    arr2 = np.ones((2,2))
    print("Q40:", arr1 * arr2)
except ValueError as e:
    print("Q40 Error:", e)

# Q41 - Calculate column-wise mean.
arr = np.array([[1,2,3],[4,5,6]])
print("Q41:", np.mean(arr, axis=0))

# Q42 - Find max value in each row.
print("Q42:", np.max(arr, axis=1))

# Q43 - Find indices of max value in each column.
print("Q43:", np.argmax(arr, axis=0))

# Q44 - Calculate moving sum along rows.
def moving_sum(arr): return np.cumsum(arr, axis=1)
print("Q44:", moving_sum(arr))

# Q45 - Check if all elements in each column are even.
arr = np.array([[2,4,6],[3,5,7]])
print("Q45:", np.all(arr % 2 == 0, axis=0))

# Q46 - Reshape array into matrix m x n.
def reshape_array(arr, m, n): return arr.reshape((m,n))
print("Q46:", reshape_array(np.arange(6), 2, 3))

# Q47 - Flatten matrix to 1D array.
def flatten_matrix(mat): return mat.flatten()
print("Q47:", flatten_matrix(np.array([[1,2],[3,4]])))

# Q48 - Concatenate two arrays along given axis.
def concat_two(arr1, arr2, axis): return np.concatenate((arr1, arr2), axis=axis)
print("Q48:", concat_two(np.array([[1],[2]]), np.array([[3],[4]]), 1))

# Q49 - Split array into sub-arrays.
def split_array(arr, parts): return np.array_split(arr, parts)
print("Q49:", split_array(np.arange(10), 3))

# Q50 - Insert and delete elements from array.
def insert_delete(arr, ins_i, ins_v, del_i):
    arr = np.insert(arr, ins_i, ins_v)
    arr = np.delete(arr, del_i)
    return arr
print("Q50:", insert_delete(np.array([1,2,3,4,5]), [2,4], [10,11], [1,3]))

# Q51 - Element-wise addition of two arrays.
arr1 = np.random.randint(1, 10, size=10)
arr2 = np.arange(1, 11)
print("Q51:", arr1 + arr2)

# Q52 - Subtract arr2 from arr1 element-wise.
arr1 = np.arange(10, 0, -1)
arr2 = np.arange(1, 11)
print("Q52:", arr1 - arr2)

# Q53 - Element-wise multiplication.
arr1 = np.random.randint(1, 10, size=5)
arr2 = np.arange(1, 6)
print("Q53:", arr1 * arr2)

# Q54 - Element-wise division of even array by 1–5.
arr1 = np.array([2, 4, 6, 8, 10])
arr2 = np.arange(1, 6)
print("Q54:", arr1 / arr2)

# Q55 - Exponentiation element-wise.
arr1 = np.arange(1, 6)
arr2 = arr1[::-1]
print("Q55:", arr1 ** arr2)

# Q56 - Count occurrences of substring in string array.
arr = np.array(['hello', 'world', 'hello', 'numpy', 'hello'])
substring = 'hello'
print("Q56:", np.char.count(arr, substring).sum())

# Q57 - Extract uppercase characters from array of strings.
arr = np.array(['Hello', 'World', 'OpenAI', 'GPT'])
print("Q57:", [''.join([ch for ch in word if ch.isupper()]) for word in arr])

# Q58 - Replace substring in string array.
arr = np.array(['apple', 'banana', 'apple pie'])
print("Q58:", np.char.replace(arr, 'apple', 'mango'))

# Q59 - Concatenate strings element-wise.
arr1 = np.array(['Hello', 'World'])
arr2 = np.array(['Open', 'AI'])
print("Q59:", np.char.add(arr1, arr2))

# Q60 - Find length of longest string.
arr = np.array(['apple', 'banana', 'grape', 'pineapple'])
print("Q60:", max(map(len, arr)))

# Q61 - Stats (mean, median, var, std) on 100 random ints.
data = np.random.randint(1, 1001, size=100)
print("Q61 Mean:", np.mean(data))
print("Median:", np.median(data))
print("Variance:", np.var(data))
print("Std Dev:", np.std(data))

# Q62 - Find 25th and 75th percentiles.
data = np.random.randint(1, 101, size=50)
print("Q62:", np.percentile(data, [25, 75]))

# Q63 - Correlation coefficient between two arrays.
a = np.random.rand(10)
b = np.random.rand(10)
print("Q63:", np.corrcoef(a, b))

# Q64 - Matrix multiplication using dot().
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("Q64:\n", np.dot(A, B))

# Q65 - Percentiles & quartiles of 50 ints.
arr = np.random.randint(10, 1001, 50)
print("Q65 - 10th, 50th, 90th percentiles:", np.percentile(arr, [10, 50, 90]))
print("Q65 - Q1 and Q3:", np.percentile(arr, [25, 75]))

# Q66 - Find index of specific element.
arr = np.array([1, 5, 8, 12, 15])
print("Q66:", np.where(arr == 8)[0][0])

# Q67 - Sort array in ascending order.
arr = np.random.randint(1, 100, size=10)
print("Q67:", np.sort(arr))

# Q68 - Filter elements > 20.
arr = np.array([12, 25, 6, 42, 8, 30])
print("Q68:", arr[arr > 20])

# Q69 - Filter elements divisible by 3.
arr = np.array([1, 2, 3, 4, 5, 6])
print("Q69:", arr[arr % 3 == 0])

# Q70 - Filter elements between 20 and 40 inclusive.
arr = np.array([10, 20, 30, 40, 50])
print("Q70:", arr[(arr >= 20) & (arr <= 40)])

# Q71 - Check byte order using dtype.byteorder.
arr = np.array([1, 2, 3], dtype=np.int32)
print("Q71:", arr.dtype.byteorder)

# Q72 - Byte swap array in-place using byteswap().
arr = np.array([1, 2, 3], dtype=np.int32)
arr.byteswap(True)
print("Q72:", arr)

# Q73 - Swap byte order without modifying original.
arr = np.array([1, 2, 3], dtype=np.int32)
new_arr = arr.view(arr.dtype.newbyteorder('S'))  # 'S' = swapped
print("Q73 Original:", arr)
print("Q73 Swapped Byte Order View:", new_arr)



# Q74 - Conditional byte swap based on system endianness.
arr = np.array([1, 2, 3], dtype=np.int32)
if arr.dtype.byteorder not in ('=', '|'):
    arr = arr.newbyteorder()
print("Q74:", arr)

# Q75 - Check if byte swap is necessary.
arr = np.array([1, 2, 3], dtype=np.int32)
needs_swap = arr.dtype.byteorder not in ('=', '|')
print("Q75:", needs_swap)

# Q76 - Modify copy array and check original.
arr1 = np.arange(1, 11)
copy_arr = arr1.copy()
copy_arr[0] = 999
print("Q76 Original:", arr1)
print("Copy:", copy_arr)

# Q77 - Modify slice (view) and check original matrix.
matrix = np.random.randint(1, 10, size=(3,3))
view_slice = matrix[0:2, 0:2]
view_slice[0, 0] = 999
print("Q77 Modified matrix:\n", matrix)

# Q78 - Broadcast addition to slice and check original.
array_a = np.arange(1,13).reshape(4,3)
view_b = array_a[1:3, :]
view_b += 5
print("Q78 Updated array_a:\n", array_a)

# Q79 - Reshape view and modify — check reflection.
orig_array = np.arange(1, 9).reshape(2, 4)
reshaped_view = orig_array.reshape(4, 2)
reshaped_view[0, 0] = 99
print("Q79:\n", orig_array)

# Q80 - Copy filtered data and check if it affects original.
data = np.random.randint(1, 10, (3,4))
data_copy = data[data > 5].copy()
data_copy[0] = 999
print("Q80 Original:\n", data)
print("Copy:", data_copy)

# Q81 - Matrix addition and subtraction.
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print("Q81 Addition:\n", A + B)
print("Subtraction:\n", A - B)

# Q82 - Matrix multiplication (3x2) * (2x4)
C = np.random.randint(1, 5, size=(3,2))
D = np.random.randint(1, 5, size=(2,4))
print("Q82:\n", np.dot(C, D))

# Q83 - Find transpose of matrix.
E = np.array([[1,2,3],[4,5,6]])
print("Q83:\n", E.T)

# Q84 - Compute determinant of square matrix.
F = np.array([[1,2],[3,4]])
print("Q84 Determinant:", np.linalg.det(F))

# Q85 - Find inverse of square matrix.
G = np.array([[1,2],[3,4]])
print("Q85 Inverse:\n", np.linalg.inv(G))
