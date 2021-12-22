import numpy as np
import math

B = np.array([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
              [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
              [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
              [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
              [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
              [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
              [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
              [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
              [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
              [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
              [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])
def H(B):
    I = np.mat(np.eye(12), dtype=int)  # единичная матрица 12х12
    H = np.concatenate((I, B))  # проверочная матрица H
    return H
def GolayMatrix(H):
    G = np.transpose(H)  # матрица расширенного кода Голея G
    return G

def Encode(G, input):# кодирование сообщения
    return (input @ G) % 2
def WH(a):  # расчет веса Хэмминга
    wt = 0
    for i in range(int(len(a))):
        if a[i] == 1:
            wt += 1
    return wt
def Decode(B, H, input):#декодирование с обнаружением ошибок
    u = [],
    zero = [0] * 12
    e = [0] * 12
    check = False
    synd = (input @ H) % 2
    ss1 = []
    for i in range(0, 12):
        ss1.append(int(synd[0, i]))
    if (WH(ss1) <= 3):#вес синдрома <=3 - синдром ошибки
        u = np.append(ss1, zero)#[синдром ошибки, 12 нулей]
        check = True
    else:
        for j in range(12):
            if (WH((ss1.copy() + B[j]) % 2) <= 2):#складываем синдром с каждой строкой B, проверяем вес
                #вес <=2 - синдром ошибки
                e[j] = 1
                u = np.append((ss1 + B[j]) % 2, e)#[сумма, нулевой вектор с 1 в позиции=номер строки B]
                check = True
                break
    if (check == False):
        synd2 = (synd@ B) % 2#второй синдром
        s3 = []
        for i in range(0, 12):
            s3.append(int(synd2[0, i]))
        if (WH(s3) <= 3):#вес синдрома <=3 - синдром ошибки
            u = np.append(zero, s3)#[12 нулей, синдром ошибки]
            check = True
        else:
            for k in range(12):
                if (WH((s3.copy() + B[k]) % 2) <= 2):#складываем синдром с каждой строкой B, проверяем вес
                #вес <=2 - синдром ошибки
                    e[k] = 1
                    u = np.append(e, (s3 + B[k]) % 2)#[нулевой вектор с 1 в позиции=номер строки B, сумма]
                    check = True
                    break
    if (check == False):
        print("Ошибку нельзя исправить (4х-кратная)")
        if (len(u) == 12):
            u = np.append(u, zero)
    else:
        print("Вектор ошибки:", u)
        if len(u)>0:
            return (input - u) % 2
        else:
            return None

def RMMatrix(r, m):#порождающая матрица
    if (r == 0):
        return np.ones(2 ** m, dtype=int)#массив 1
    if (r == m):
        bot = np.zeros(2 ** m, dtype=int)#массив 0
        bot[len(bot) - 1] = 1#последний эл-т 1
        res = RMMatrix(m - 1, m)#рекурсивный вызов
        return np.vstack((res, bot))
    t = np.concatenate((RMMatrix(r, m - 1), RMMatrix(r, m - 1)), axis=1)#верхняя строка матрицы
    b = RMMatrix(r - 1, m - 1)#правая нижняя
    if (len(b.shape) == 1):
        zeros = np.zeros(len(t[0]) - len(b), dtype=int)
        b = np.append(zeros, b)
    else:
        zeros = np.zeros((len(b), (len(t[0]) - len(b[0]))), dtype=int)
        b = np.concatenate((zeros, b), axis=1)
    return np.vstack((t, b))

def RMHMatrix(i, m):#проверочная матрица
    I1 = np.eye(2 ** (m - i), dtype=int)#левая единичная матрица
    I2 = np.eye(2 ** (i - 1), dtype=int)#правая единичная матрица
    H = np.array([[1, 1], [1, -1]], dtype=int)
    for i in range(len(I1)):
        H_n = I1[i][0] * H
        for j in range(1, len(I1)):
            H_n = np.concatenate((H_n, I1[i][j] * H), axis=1)
        if (i == 0):
            H2 = H_n
        else:
            H2 = np.concatenate((H2, H_n))
    for i in range(len(H2)):
        H_n = H2[i][0] * I2
        for j in range(1, len(H2)):
            H_n = np.concatenate((H_n, H2[i][j] * I2), axis=1)
        if (i == 0):
            result = H_n
        else:
            result = np.concatenate((result, H_n))
    return result

def RMEncode(G, input): # кодирование сообщения
    return (input @ G) % 2

def RMDecode(input): # декодирование
    inp = input.copy()
    for i in range(len(inp)):
        if (inp[i] == 0):
            inp[i] = -1# во входящем сообщении меняем 0 на -1
    R1 = RMHMatrix(1, M)#вычисление H
    w1 = (inp @ R1)
    wi = np.array([w1, (w1 @ RMHMatrix(2, M))])
    for i in range(2, M):
        wi = np.vstack((wi, (wi[i - 1] @ RMHMatrix(i + 1, M))))
    max_value = max(wi[len(wi) - 1], key=abs)#макс.по абс.зн-ю компонент
    for i in range(len(wi[0])):
        if (wi[len(wi) - 1][i] == max_value):
            break
    j = bin(i)#двоичное представление
    j = [int(x) for x in list(j[2:len(j)])]
    if (len(j) < n - k - 1):
        j = np.append(np.zeros(n - k - 1 - len(j), int), j)
    j = np.flipud(j)
    if (max_value > 0):#j-ый компонент положительный
        j = np.append(1, j)#исходное сообщение
    else:#j-ый компонент отрицательный
        j = np.append(0, j)
    return j[:k]


H = H(B)
print("H:\n", H)
G = GolayMatrix(H)
print("\nG:\n",G)

mas =np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1])#тестовый пример сообщения
cod = Encode(G,mas)#кодирование
cod = np.squeeze(np.asarray(cod))
ishod = cod.copy()
print("Закодированное сообщение:",cod)

#кол-во ошибок от 1 до 3
while True:
    count_er = int(input("Количество ошибок?: "))
    if not 1 <= (count_er) <= 4:
        print("Попробуйте снова")
    else:
        print(count_er)
        break
mas_er=[]
z=0
#последовательно вносим ошибки
while z < count_er:
    i = int(input("В какой бит внести ошибку?: "))
    if not 0 <= (i) < 24: #не выходим за границы сообщения
        print("Число не в диапазоне, попробуйте снова")
        i = int(input("В какой бит внести ошибку?: "))
    elif i in mas_er:
        print("Число ранее было задано,попробуйте снова")#е меняем один и тот же бит несколько раз
        i = int(input("В какой бит внести ошибку?: "))
    else:
        print("i =", i)
        mas_er.append(i)#запоминаем номер бита для проверки
        cod[i] = not cod[i]#еняем бит
        print("Cлово с ошибкой в бите",i,":", cod)
    z+=1
decod = Decode(B,H,cod)
print("Декодированное сообщение:", decod)
if(np.array_equal(ishod, decod)):
    print("Ошибки были исправлены корректно")

##### ЧАСТЬ 2 ########

while True:
    R = int(input("Введите r от 1 до 3: "))
    if not 1 <= (R) <= 3:
        print("Попробуйте снова")
    else:
        print("r =", R)
        break
while True:
    M = int(input("Введите m от 1 до 4: "))
    if not 1 <= (M) <= 4:
        print("Попробуйте снова")
    elif M<R:
        print("Попробуйте снова")
    else:
        print("m =", M)
        break

#расчет размерностей
k = 0
for i in range (R + 1):
    k = k + int(math.factorial(M)/(math.factorial(i)*math.factorial(M-i)))
n = 2**M

#вызов функций
RMG = RMMatrix(R,M)
RMH = RMHMatrix(R,M)

print("RM Matrix:\n", RMG, "\nRM H:\n", RMH)

mas2 = []
print("Введите", k, "символов")
mas2 = [int(input()) for i in range(k)]#ввод сообщения
encod = RMEncode(RMG, mas2)#кодирование
encod = np.squeeze(np.asarray(encod))
ishod2 = encod.copy()
print("Закодированное сообщение:",encod)

#кол-во ошибок от 1 до 3
while True:
    count_er = int(input("Количество ошибок?: "))
    if not 1 <= (count_er) <= 4:
        print("Попробуйте снова")
    else:
        print(count_er)
        break
mas_er2=[]
z=0
#последовательно вносим ошибки
while z < count_er:
    i = int(input("В какой бит внести ошибку?: "))
    if not 0 <= (i) < n: #не выходим за границы сообщения
        print("Число не в диапазоне, попробуйте снова")
        i = int(input("В какой бит внести ошибку?: "))
    elif i in mas_er2:
        print("Число ранее было задано,попробуйте снова")#е меняем один и тот же бит несколько раз
        i = int(input("В какой бит внести ошибку?: "))
    else:
        print("i =", i)
        mas_er2.append(i)#запоминаем номер бита для проверки
        encod[i] = not encod[i]#еняем бит
        print("Cлово с ошибкой в бите",i,":", encod)
    z+=1

decod2 = RMDecode(encod)
print("Декодированное сообщение:", decod2)
