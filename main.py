import re
import sys
from functools import partial, reduce
from itertools import repeat, takewhile, groupby, tee, zip_longest, islice, product, starmap, dropwhile, accumulate
from math import floor
from operator import mul, truediv, sub, eq, gt, lt, ge, itemgetter as item, methodcaller, floordiv, mod, add
from toolz import compose, identity, cons
from random import random
from statistics import mean
from toolz import compose, juxt, partition, curry, flip, remove, take_nth, drop, sliding_window, concat, identity, topk, \
    get
from itertools import chain


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import csv

# ================== Введение в Python. Практикум ========================

# def paths_count(x, y):
#     r =[]
#     c = 1
#     if x > 0:
#         r += [(x-1,y)]
#     if y > 0:
#         r += [(x,y-1)]
#     if r:
#         c = 0
#         for t in r:
#             c += paths_count(*t)
#     return c


# print(*accumulate([(1,0),(0,1)],partial(map,sub)))

# def first(*args,**kwargs):
#     if args:
#         return args[0]
#     if kwargs:
#         return kwargs[sorted(kwargs.keys())[0]]
#     return None

# def call(fun, *args,**kwargs):
#     return fun(*args, **kwargs)

# def header(text,level=1):
#     return f"{'#'*(level>6 and 6 or level if level else 1)} {text}"

# def add(a, b):
#     return a + b

# def f_map(func, l):
#     return map(func, l)

# def header(text,level=1):
#     return f"{'#'*(level>6 and 6 or level if level else 1)} {text}"

# def percent(share, round_digits=None):
    # return f"{round_digits and round(share*100,round_digits) or int(share*100)}%"
    # return f"{round(share*100,round_digits)}%"

# def concat(*str,sep=' '):
#     return sep.join(str)

# from operator import concat as conc
# from functools import reduce
# def concat(*str):
#     return reduce(conc, str)

# def sq_sum(*numbers):
#     return sum(map(flip(pow)(2), numbers))

# def mean(*numbers):
#     return sum(numbers)/len(numbers)


# def func(l):
#     res =[]
#     for i in l:
#         if i not in res:
#             res.append(i)
#     return res

# print(func([1, 2, 2, 3, 4, 3, 6, 2]))

# def func(l1, l2):
#     res = l1[::]
#     for i in l2:
#         if i in res:
#             res.remove(i)
#     return res

# print(func([1, 3, 2, 4, 2], [2, 5]))
# def is_prime(a):
#     return all(map(lambda x: a % x, range(2, int(a**0.5+1))))

# def increase_g():
#     pass
#
# def fahrenheit_to_celcius(degree):
#     return (degree - 32) *5/9
#
# def celcius_to_fahrenheit(degree):
#     return degree *9/5 +32
#
# def multiply(a, b):
#     return a*b
#
# def add_2(a):
#     return a+2

# input = 'l r r * d d # r # l l l d'
# d = {'l':(-1,0,0),'u':(0,1,0),'d':(0,-1,0),'r':(1,0,0),'*':(0,0,1),'#':(0,0,-1)}
# c = map(lambda x:d[x], input.split())
# res = reduce(lambda a,x: a[2]!=0 and tuple(partial(map, add)(a,x)) or a, c, (0,0,1))
# res = reduce(partial(map, add), takewhile(lambda a:a[2],c), (0,0,1))
# print('{} {}\n{}'.format(*res))
# res = reduce(lambda a,x: a[2] and tuple(partial(map, add)(a,x)) or a, c, (0,0,1))
# acc = takewhile(lambda a: a[2]>=0, accumulate(compose(tuple, partial(map,add)), c, (0,0,1)))
# print('{} {}\n{}'.format(*res))

# f = map(curry(flip(get))(d), input.split())
# print(*f)

# print(tuple(partial(map,add)((1,2,3),(1,1,1))))

# a = reduce(partial(map,add),)

# op = {'-':sub,'+':add,'*':mul,'/':truediv}
# l = map(lambda x:curry(flip(op[x[0]]))(float(x[2:])), iter(input,'.'))
# print(reduce(lambda a, f: f(a), list(l), float(input())))


# op = {'-':sub,'+':add,'*':mul,'/':truediv}
# l = list(map(compose(lambda x:curry(flip(op[x[0]]))(float(x[1])), methodcaller('split')), iter(input,'.')))
# print(reduce(lambda a, f: f(a), l, float(input())))

# n = int(input())
# map(lambda x:)
#
# print(*l,n)




# l = iter(input,'.')
# res = list(map(lambda x:x[:-5], filter(lambda x:'true' in x,l)))
# print(*res)
# print(len(res))

# print(*map(lambda x:x[-4:]=='true' and x[:-5],l))


# n = 4
# print(*map(lambda x: ' '*(n-x-1)+'#'*(1+x*2), range(n)),sep='\n')

# i = 'camelCase'
# print(''.join(map(lambda x: x.isupper() and '_'+x.lower() or x, list(i))))
# print(''.join(map(lambda x: ord(x) in range(65,91) and '_'+chr(ord(x)+32) or x,list(i))))

# 97 - 122 a-z
# 65 - 90 A-Z
# print(ord(i))
# print(ord(i) in range(65,91) and chr(ord(i)+32) or i)

# i = 'snake_case'
# l = i.split('_')
# for i in range(1,len(l)):
#     l[i] = l[i].capitalize()
# print(''.join(l))

# h = """ .-.  .-.
# |   \/   |
# \        /
#  `\    /`
#    `\/`   """
#
# k = int(input())
# for i in h.split('\n'):
#     print(i*k)


# a = input
# print(a.isnumeric() and int(a)*2 or a*2)
# print(*reversed(list(map(int,input.split()))))
# """
# в пустой (мёртвой) клетке, рядом с которой ровно три живые клетки, зарождается жизнь;
# если у живой клетки есть две или три живые соседки, то эта клетка продолжает жить;
# в противном случае, если соседей меньше двух или больше трёх, клетка умирает
# """

# def live_or_die(i,j):
#     c = 0
#     for k,l  in [(1,0),(1,1)]:
#         for n in range(4):
#             if -1 < i+k <10 and -1 < j+l <10:
#                 c += init[i+k][j+l] == '#'
#             k, l = -l, k
#     return '.#'[init[i][j]=='.' and c == 3 or init[i][j]=='#' and 1 < c < 4]

# init = [['.', '#', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '#', '.', '.', '.', '.', '.', '.', '.'],
#         ['#', '#', '#', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
#         ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']]

# for _ in range(15):
#     init = [[live_or_die(i,j) for j in range(10)] for i in range(10)]
#
# print(*init,sep='\n')

# print(init[i][j],end='')
# print()

# init = '#....#'
# elem = lambda seq: map(lambda x: x=='#','.'+seq+'.')
# live_or_die = lambda x: (x[0] ^ x[2]) or (not x[1] and x[0] and x[2])
# for _ in range(int(input())):
#     init = reduce(lambda a, x: a+'.#'[live_or_die(x)], sliding_window(3,elem(init)),'')
#
# print(list(init))

# lst = list(map(int,iter(input,'.')))
# print(list(filter(lambda x: x<3,lst))+list(filter(lambda x:x>=3,lst)))
# n = int(input())
# print(*[['#']*n]*n,sep='\n')

# print(list(concat(juxt(curry(filter)(lambda x: x<3),curry(filter)(lambda x:x>=3))(list(map(int,iter(input,'.')))))))
# print(*map(juxt(curry(filter)(lambda x: x<3),curry(filter)(lambda x:x>=3)),tee(map(int,iter(input,'.')),2)))

# lst = list(map(int,iter(input,'.')))
# lst2 = topk(3,lst)
# # print(list(concat([reversed(lst2),filter(lambda x: x not in lst2, lst)])))
# for i in lst2:
#     lst.remove(i)
# print(list(lst2)+lst)

# lst = list(map(int,iter(input,'.')))
# m = max(lst)
# lst.remove(m)
# lst.insert(0,m)
# print(lst)

# print(list(filter(identity,concat(zip_longest(*[input() for _ in range(2)])))))
# print(list(map(compose(flip(truediv)(2),sum),sliding_window(2,map(int,input)))))
# print(list(accumulate(map(int,input),compose(flip(truediv)(2),add))))
# print(truediv(*reduce(partial(map,add),zip(input,repeat(1)))))
# print(list(map(int,input)))


# print(pow(sum(map(curry(flip(pow)(2)), coord)), 0.5))
# print(("NO","YES")[float(input())>pow(sum(map(curry(flip(pow)(2)), coord)), 0.5)])
# print(("NO","YES")[eq(*partition(2,map(curry(lt)(0),[float(input()) for _ in range(4)])))])
# print('1423'[int(''.join(map(lambda x:str(int(x<0)),coord)),2)])
# print(truediv(*reduce(partial(map,add),zip(map(int,iter(input, '.')), repeat(1)), (0,0))))
# print(sum(map(curry(flip(pow))(2),map(int,iter(input, '.')))))
# print(*map(curry(pow)(2),map(float,['3','4'])))

# print(max(map(float, iter(input, '.'))))
# print(sorted(map(float, iter(input, '.')))[-2])

# print(max(remove(lambda x: x==max(seq),seq)))
# print(sum(take_nth(2, map(float, iter(input, '.')))))
# fib = list(takewhile(curry(ge)(n), map(item(0), accumulate(repeat((0, 1)), lambda a, _: juxt(item(1), sum)(a)))))
# print(reduce(lambda a,_:juxt(item(1),sum)(a), range(8),(0,1)))
# print(("prime", "composite")[any(map(lambda x:not a%x, range(2,a)))])
# print(compose(sum,partial(map,curry(flip(pow))(2)))(range(1,6)))
# print(sum(filter(lambda x:x%2,range(1,2*2))))
# print(sum(range(1,int(input())+1)))
# a = int(input())
# print(*[a] if a%3 and a%5 else ['Fizz','Buzz'][1-(not a%3):1+(not a%5)])
# print(sorted(map(int, [input() for _ in range(3)])), sep='\n')
# print(('BLACK','WHITE')[(x-1)%2 ^ (y-1)%2])
# coord = partition(2,list(map(int, input.split())))
# print(("NO", "YES")[max(partial(map, compose(abs,sub))(*coord))==1])
# print(max(map(int,[input() for _ in range(3)])))

# s = input()
# if int(s[-1]) in [0,range(5, 10)] or len(s)>1 and int(s[-2]) == 1:
#     print(s,"studentov")
# elif int(s[-1]) in range(2,5):
#     print(s,"studenta")
# else:
#     print(s,"student")

# print(['zero','one','two','three','four','five','six','seven','eight','nine'][int(input())])

# s = [int(input()) for _ in range(2)]
# print(0 if s == s[::-1] else s.index(max(s))+1)

# s = int(input())
# print(0 if s == 0 else s//abs(s))
# =========================================================================


# ==================Python. Functional Programming=========================


# from toolz import compose, curry
# from operator import add, mul, eq, methodcaller, itemgetter
# # from itertools import chain, tee, count, cycle, repeat, islice, takewhile, dropwhile, starmap, groupby, product
#
# take = lambda n, it: islice(it, 0, n)
# drop = lambda n, it: islice(it, n, None)
#
# head = partial(take, 1)
# tail = partial(drop, 1)
#
# iterate = lambda f, x: accumulate(repeat(x), lambda a, b : f(a))
# zipWith = lambda tfunc: compose(partial(map, tfunc), zip)
# concat = lambda seq: chain(*seq)
# concatMap = compose(concat, map)
#
# seq = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
# seq = [5, 4, 32, 32, 4]

# print(*tail(seq))
# print(*product(range(4),range(5)))
# print(*starmap(add,product(range(4),range(5))))
# def uniquify(seq):
#     for k,v in groupby(seq):
#         yield k

# def uniquify(seq):
#     prev = None
#     for i in seq:
#         if i != prev:
#             yield i
#             prev = i


# iterate = lambda f, x: accumulate(repeat(x), lambda a, _: f(a))

# nextStep = lambda p: (p[1], p[0] % p[1])
# euqSeq = lambda a, b: takewhile(item(1), iterate(nextStep, (a, b)))
#
# euqSeq = lambda a, b: takewhile(lambda t: t[1], iterate(lambda t: (t[1], t[0] % t[1]), (a, b)))

# itarate(lambda a: sum(*a),(0,1) )

# z = takewhile(lambda _: True, iterate(lambda t: (t[1],t[0]+t[1]), (0, 1)))
# fibSeq =lambda :map(item(1), iterate(juxt(item(1), sum), (0, 1)))

# def euqSeq(a, b):
#     if b:
#         yield from cons((a, b), euqSeq(b, a % b))

# it = euqSeq(78, 2064)
# print(list(it))
# print(list(accumulate(add, [1, 1, 2, 3, 4, 5])))
# print(list(accumulate(add,chain(repeat(1,3), range(2,5)))))

# mapId = compose(list, partial(map, identity))
# filterId = compose(list, partial(filter, lambda x: True))
# reduceId = compose(list,partial(reduce,lambda a,b: a+list(b),[]))


# print(partial(reduce, lambda a,b: a+[b],[])([1,2,34]))

# print(*reduce(partial(map,add),zip(range(5),repeat(1)),(0,0)))
# print(reduceId([1,2,3,4,5]))
# =========================================================================

# =======================Python для решения практических задач==============
#
# from http.server import HTTPServer, CGIHTTPRequestHandler
# server_address = ("", 8000)
# httpd = HTTPServer(server_address, CGIHTTPRequestHandler)
# httpd.serve_forever()

# from lxml import etree
#
# html = etree.Element("html")
# table = etree.SubElement(etree.SubElement(html,'body'),'table')
#
# for i in range(1,11):
#     row = etree.SubElement(table,'tr')
#     for j in range(1,11):
#         a = etree.SubElement(etree.SubElement(row, 'td'), 'a')
#         val = str(i*j)
#         a.set('href', f'http://{val}.ru')
#         a.text = val
# with open('l_5_1_step5.html','wb') as outf:
#     outf.write(etree.tostring(html, pretty_print=True))

# print(str(etree.tostring(html, pretty_print=True),'utf-8'))


# from itertools import repeat, groupby
# from functools import partial, reduce
# from operator import add, mul, itemgetter as item
# from toolz import drop
# from math import floor
# import xlrd
#
# wb = xlrd.open_workbook('trekking3.xlsx')
# getvalue = lambda cell: 0 if isinstance(cell.value, str) else cell.value
# catalog = {row[0].value:list(map(getvalue,row[1:5])) for row in drop(1, wb.sheet_by_index(0).get_rows())}
# days = groupby(drop(1, wb.sheet_by_index(1).get_rows()), lambda row:row[0].value)
# itemval = lambda o:partial(map,mul)(repeat(o[2].value/100),catalog[o[1].value])
#
# for day in map(item(1),days):
#     print(*map(floor,reduce(partial(map,add),map(itemval,day),(0,0,0,0))))


# from lxml import etree
#
# tree = etree.parse('map2.osm')
# root = tree.getroot()
# print(len(root.xpath('//tag[@k="amenity" and @v="fuel"]')))
# print(c)


# import xmltodict
#
# fin = open('map1.osm', 'r', encoding='utf8')
# xml = fin.read()
# fin.close()
#
# parsedxml = xmltodict.parse(xml)
# print(parsedxml['osm']['node'][100]['@id'])


# with zipfile.ZipFile('rogaikopyta.zip','r') as arch, open('ved.txt','w',encoding='utf-8') as outf:
#     res = []
#     for info in arch.infolist():
#         print(f'reading {info.filename}', end=' ')
#         wb = xlrd.open_workbook(file_contents=arch.read(info.filename))
#         sh = wb.sheet_by_index(0)
#         print(sh.cell_value(1,1),str(sh.cell_value(1,3)))
#         res += [(sh.cell_value(1,1),str(int(sh.cell_value(1,3))))]
#
#     outf.write('\n'.join(map(' '.join,sorted(res))))

# resp = urlopen('https://stepik.org/media/attachments/lesson/245267/salaries.xlsx')
#
# with open('salaries.xlsx','wb') as outf:
#     outf.write(resp.read())

# wb = xlrd.open_workbook('salaries.xlsx')
# resp = requests.get('https://stepik.org/media/attachments/lesson/245267/salaries.xlsx')
# wb = xlrd.open_workbook(file_contents=resp.content)
# wb = xlrd.open_workbook('trekking2.xlsx')
# sh = wb.sheet_by_index(0)
# d = {sh.cell_value(row, 0): [0 if isinstance(i,str) else i for i in sh.row_values(row)[1:]] for row in range(1,38)}
# sh = wb.sheet_by_index(1)
# e = [list(map(curry(mul)(sh.cell_value(row, 1)/100),d[sh.cell_value(row, 0)])) for row in range(1,13)]
#
# print(*map(compose(floor, sum), zip(*e)))


# map(mul, d[sh.cell_value(row, 0)])


# print()
# for k,v in e.items():
#     print(k,v)

# print(list(map(curry(mul)(2),d[sh.cell_value(row, 0)])))

# print(float('205,8'.replace(',','.')))
# print(sorted(sh.row_values(1)[1:]))
# print(median(sh.row_values(1)[1:]))
# print(sh.col_values(1)[1:])
# for rownum in range(7, 27):
#     temp = sh.row_values(rownum)
#     nmin = min(nmin, temp[2])
# print(nmin)


# def sampleCount(s):
#     return lambda l: sum(map(partial(flip(str.count), s), l))
# sampleCount = lambda s:lambda l: reduce(lambda a, b: a + b.count(s), l, 0)
#
# from urllib.request import urlopen, urlretrieve
# from bs4 import BeautifulSoup
#
# resp = urlopen('https://stepik.org/media/attachments/lesson/209723/5.html') # скачиваем файл
# html = resp.read().decode('utf8') # считываем содержимое
# soup = BeautifulSoup(html, 'html.parser') # делаем суп
#
# print(sum(map(lambda tag:int(tag.text), BeautifulSoup(html, 'html.parser').find_all('td'))))

# table = soup.find('table')
# cn = 0
# for tr in soup.find_all('tr'):
#    for td in tr.find_all('td'):
#        print(td.text)
#
# print(cn)
# with open('2.html','r',encoding='utf-8') as text:
#     code = [x for line in text for x in re.findall(r'(?<=<code>).*?(?=</code>)', line)]
#
# d = {c: code.count(c) for c in code if code.count(c)>1}
# k, m = max(d.items(), key=lambda kv: kv[1])
# print(*[k for k,v in sorted(d.items(), key=lambda kv:kv[0]) if v == m])
# ========================================================================

# ==================Python. Functional Programming=========================
# you are only asked to implement functions myAll, myAny, elem
# myAll takes a predicate and a sequence, returns True if all the elements of the sequence satisfy the given predicate and False otherwise
# myAny  takes a predicate and a sequence, returns True if there is an element of the sequence that satisfies the given predicate

# myAll = compose(bool, lambda s: reduce(__and__,s), map)
# myAny = compose(bool, lambda s: reduce(__or__,s), map)
# elem = lambda n: lambda seq: myAny(lambda x: __eq__(x, n), seq)
# elem = compose(curry(myAny), curry(__eq__))


# implement a function elem that takes an element
# and returns another function.
# This last function takes a sequence and returns
# True if the given element is in this sequence
# False otherwise.
# Implement the elem function in terms of myAll or  myAny.

# containsFive = elem(5)
# containsFive([1, 2, 3, 4])
# False
# elem(2)((1, 2, 3, 4))
# True
# print(elem(1)((1, 2, 3, 4)))


# odd = lambda x: x % 2
# print(myAll(odd,(2,5,7)))

# maximum = lambda x: reduce(max, x, -inf)

# print(maximum(a))
# raise ValueError()

# def pyTriple(x):
#     for m in range(1, x):
#         for n in range(1, m):
#             yield m**2-n**2, 2*n*m, n**2+m**2


# def findTheTriple():
#     sqf = lambda func:lambda m,n: func(pow(m,2),pow(n,2))
#     abc = lambda x: (sqf(sub)(*x), 2*mul(*x), sqf(add)(*x))
#     return set(*filter(lambda x:sum(x)==1000, map(abc, filter(lambda x: gt(*x), product(range(1,25),repeat=2)))))


# print(findTheTriple())
# print(*product(range(1,5),repeat=2))

# print([i for i in pyTriple(25) if sum(i) == 1000])

# text = urlopen('https://stepik.org/media/attachments/lesson/209717/1.html').read().decode('utf-8')
# print('C++' if text.count('C++')>text.count('Python') else 'Python')


# zipWith = lambda tfunc: compose(partial(map, tfunc), zip)

# ( sum(x*y - awg(x)*awg(y)) / n ) / sigma(x) * sigma (y)
# awg = lambda x:  sum(x)/len(x)
# sigma = lambda x: sqrt(reduce(lambda a, b: a + (b - awg(x)) ** 2, x, 0) / len(x))

# divPair = lambda t: truediv(*t)
# mean = lambda seq: truediv(*reduce(partial(map, add), zip(seq, repeat(1)), (0, 0)))
# [input() for _ in range(2)

# compose(lambda x: mul(*x), map)(int, [2,3])
# x, y = map(int,[input() for _ in range(2)])
# a,b,c,d = map(int,[input() for _ in range(4)])
# print(c*d + (b+d)*a)


def crlt(x, y):
    tfunc = lambda a: mul(*a) - reduce(mul, map(awg, (x, y)))
    return (sum(zipWith(tfunc)(x, y)) / len(x)) / (sigma(x) * sigma(y))


def overprice(rest, perc, n, mes=1):
    ov = rest * perc / 100 / 12, 1
    if rest - n + ov[0] > 0:
        print(f'{mes} проценты:{ov[0]:.2f} тело:{n - ov[0]:.2f} долг:{rest - n + ov[0]:.2f}')
        ov = partial(map, add)(overprice(rest - n + ov[0], perc, n, mes + 1), ov)
    else:
        print(f'{mes} проценты:{ov[0]:.2f} тело:{rest - ov[0]:.2f} долг:0.00')
    return ov


# tot, m = overprice(5625000,8.5,45337)
# tot, m = overprice(1000000,11.7, 22300)
# print(f'итого переплата {tot:.2f} в сумме {tot+5625000:.2f}')


def mean(x):
    s, n = reduce(partial(map, add), zip(x, repeat(1)), (0, 0))
    return s / n


def var(x):
    s, n = reduce(partial(map, add), zip(x, repeat(1)), (0, 0))
    return reduce(lambda a, b: a + (b - s / n) ** 2, x, 0) / n


def stdv(x):
    return var(x) ** 0.5


f = lambda x: x + 1
x = 0


# succ = lambda n: lambda s: lambda z: s(n(s)(z))
# succ2 = lambda n: lambda s: lambda z: n(s)(s(z))
# toint = lambda n: n(lambda x: x+1)(0)
# add = lambda n: lambda m: m(succ)(n)
# mul = lambda n: lambda m: m(add(n))(zero)
# pow = lambda n: lambda p: p(mul(n))(one)
# print(toint(mul(three)(two)))
# print(pow(two)(three)(f)(x))
# tobool = lambda b: b(1)(0)
# zero = lambda s: lambda z: z
# one = lambda s: lambda z: s(z)
# two = lambda s: lambda z: s(s(z))
# three = lambda s: lambda z: s(s(s(z)))
#
# one = lambda f: lambda x: f(x)
#
# true = lambda x: lambda y: x
# false = lambda x: lambda y: y
#
#
# def isZero(numeral):
#     return numeral(lambda _: false)(true)


# und = lambda b1: lambda b2: b1(b2)(b1)
# решение 2.14
# orr = lambda b1: lambda b2: b1(b1)(b2)


# print(und(false)(true)(1)(0))


# решение 2.13
# pow = lambda m: lambda n: n(m)
# неправильное
# pow = lambda m: lambda n: lambda f: lambda x: m(f)(x) ** n(f)(x)

# print(pow(two)(three)(f)(x))
# print(pow(three)(two)(f)(x))


# решение 2.12
# def makeDecorator(subdecorator):
#     return lambda func: lambda *a, **k: subdecorator(func, *a, **k)


# def makeDecorator(subdecorator):  # introduce
#     def inner(func):  # id
#         def inner2(*a, **k):
#             return subdecorator(func, *a, **k)
#         return inner2
#     return inner


# print(*(id(40, 2, key=12345)))


#  решение 2.11
# def flip(f):
#     return lambda *args: f(*args[::-1])


# bucket = lambda *args, **kwargs: lambda f: lambda *ar, **kw: (args, kwargs, f(*ar, **kw))

# def bucket(*ar, **kw):
#     def decorator(func):
#         # @wraps(func)
#         def inner(*arg, **kwargs):
#             return ar, kw, func(*arg, **kwargs)
#         return inner
#     return decorator


# class ImDict(dict):
#     def _immutable(self, *args, **kws):
#         raise TypeError('objects of type ImDict are immutable')
#
#     __setitem__ = _immutable
#     __delitem__ = _immutable
#     clear = _immutable
#     update = _immutable
#     setdefault = _immutable
#     pop = _immutable
#     popitem = _immutable
#
#
# class SwitchDict(dict):
#     def petrify(self):
#         return ImDict(self)

# ages = [1, 2, 3, 4, 5]
# print(zip(ages, repeat(1)))
# print(*functools.reduce(functools.partial(map, add), zip(ages, repeat(1)), (0, 0)))


# reduce(partial(map, add), zip(ages, repeat(1)), (0, 0))

# def flatten(d):
#     n = lambda key,val: if isinstance(val,dict)
#     for k, v in d.items():
#         if isinstance(v, dict):
#             return flatten(v)
#
# def flatten(d):
#     def helper(d,*parent_key):
#         if isinstance(d, dict):
#             for k, v in d.items():
#                 yield from helper(v, *parent_key, k)
#         else:
#             yield ".".join(parent_key), d
#
#     return dict(helper(d))


# def deepReverse(coll):
#     if type(coll) == list:
#         coll.reverse()
#         for i in coll:
#             deepReverse(i)

# def deepReverse(coll):
#     return list(map(deepReverse, coll[::-1])) if isinstance(coll, list) else coll
#
# def timeit(func):
#     @functools.wraps(func)
#     def newfunc(*args, **kwargs):
#         startTime = time.time()
#         func(*args, **kwargs)
#         elapsedTime = time.time() - startTime
#         print('function [{}] finished in {} ms'.format(
#             func.__name__, int(elapsedTime * 1000)))
#
#     return newfunc


# @timeit
"""
Write a recursive procedure that, given an amount and values available,
returns a number of ways in which this amount can be composed using these values.
Use the following principle.

Looking at some particular value from the pool, we may define the number of
ways to compose the given amount as the sum of:
the number of ways to compose the amount without using this value
the number of ways to compose (amount - value) with using this value

Think of why it is true, define some edge-cases, implement a procedure.
Expected behaviour: 

>>> values = (1, 2, 3, 4)
>>> makeAmount(4, values)
5"""

def makeAmount(a, values):
    # return sum(makeAmount(x, values[1:]) for x in range(a, -1, -values[0])) if values else not a
    s = 0
    if values:
        print(f'begin loop  in range {list(range(a, -1, -values[0]))} with step {-values[0]}')
        for x in range(a, -1, -values[0]):
            print(f'run makeAmount for {x} in {values[1:]}')
            s += makeAmount(x, values[1:])
    else:
        s = not a
    print(f'makeAmount finished s={s} a={a} {not a}\n')
    return s

    # comb = lambda n: reduce(lambda c, el: c + 1, filter(lambda t: sum(t) == a, combinations_with_replacement(filter(lambda x: x <= a, values), n)), 0)
    # comb = lambda n: list(filter(lambda t: sum(t) == a, combinations_with_replacement(values, n)))
    # res = sum(map(comb, range(1, a//min(values)+1)))
    # res = list(map(comb, range(1, a)))
    # return res

    # def comb(n):
    #     l = list(filter(lambda t: sum(t) == a, combinations_with_replacement(filter(lambda x: x <= a, values), n)))
    #     return l if n <= 1 else l + comb(n - 1)
    #
    # return reduce(lambda c, el: c+1 if sum(el) == a else c, comb(a//min(values)),0)

# startTime = time.time()
# print(f'run makeAmount for {6} in {[2,1,3]}')
# print(makeAmount(6, [2,1,3]))
# elapsedTime = time.time() - startTime
# print('function finished in {} ms'.format(int(elapsedTime * 1000)))

# def isPalindrome2(word):
#     if len(word) < 2 or word[0] == word[-1]:
#         if (len(word)>3):
#             return isPalindrome(word[1:-1])
#         return True
#     return False


# def quickPower(base, power):
#     if power:
#         if power % 2:
#             return base * quickPower(base, power - 1)
#         return quickPower(base * base, power // 2)
#     return 1


# def meanAge2(records):
#     has_age = lambda r: 'age' in r.keys()
#     get_age = lambda r: r['age']
#     ages = map(get_age, filter(has_age, records))
#     sum_ages, total_count = reduce(partial(map, add), zip(ages, repeat(1)), (0, 0))
#     if total_count:
#         return sum_ages / total_count

# from functools import reduce
#
# def composition(*args):
#     return lambda val: reduce(lambda x, f: f(x), reversed(args), val)
# =================================================================================


# with open('input.txt', 'r') as input:
#     n, m = map(int, input.readline().split())
#     # a = [[ch for ch in input.readline().rstrip()] for _ in range(n)]
#     a = [input.readline().rstrip() for _ in range(n)]
#
# print(*a, sep='\n')
#
# def isalive(i, j):
#     neib = [a[(i + k) % n][(j + l) % m] for k in range(-1, 2) for l in range(-1, 2) if k != 0 or l != 0].count('X')
#     if a[i][j] == 'X' and (neib < 2 or neib > 3):
#         return '.'  # Клетка умирает, если число ее живых соседей не равно 2 или 3.
#     if a[i][j] == '.' and neib == 3:
#         return 'X'  # В клетку зарождается жизнь, если число ее живых соседей ровно 3
#     return a[i][j]
#
#
# [print(isalive(i, j), end='' if j < m - 1 else '\n') for i in range(n) for j in range(m)]

# print()
# print(*b)
# [print(*line, sep='')for line in b]

# print(f'for{(i,j)} cell is {a[i][j]} will be {isalive(a,i,j)}')


# a ='my_first_class'
#
# print(input().title().replace('_', ''))
# import re
# a, b = 'abcdef', 'abcdef' #input(), input()
# c = [m.start() for m in re.finditer(rf'(?=({b}))', a)]
# print(*c if any(c) else [-1])
# c = {a.find(b, i) for i in range(len(a)) if b in a[i:]}
# print(*sorted(c) if c else [-1])


# frac = lambda num, de: num % de > 0 and [num // de] + frac(de, num % de) or [num // [de, 1][num < de]]
# print(*frac(*list(map(int, input().split('/')))))

# f = lambda a,b:a+b
# def frac(num, de):
#     if num % de != 0:
#         print(num // de)
#         frac(de, num % de)
#     else:
#         print(num // [de, 1][num < de])


# input = '12/3'
# input = '239/30'
# print(frac(*list(map(int, input.split('/')))))
# frac(*list(map(int, input.split('/'))))


# def collatz_gen(n):
#     yield n
#     while n > 1:
#         n = n // 2 if n % 2 == 0 else n * 3 + 1
#         yield n
#
# print(*collatz_gen(17))

# import re
# print('3ab4c2CaB')
# input = 'aaabccccCCaB'
#
# def pack_gen(a):
#     el = a[0]
#     i = c = 1
#     while i < len(a):
#         if el != a[i]:
#             yield str(c)+el if c > 1 else el
#             el = a[i]
#             c = 0
#         c += 1
#         i += 1
#     yield str(c)+a[i-1] if c > 1 else a[i-1]


# print("".join(list(pack_gen(input))))
# print(list(range(5)))
# def pack_gen():
#     i = 0
#     m = True
#     while m:
#         m = re.search(r'^(\w)\1*', a[i:])
#         if m:
#             beg, end = m.span()
#             yield (end, a[i])
#             i += end
#
# print([str(i)+el for i, el in pack_gen()])

#
# f = lambda m: (len(m) > 1 and m[0] == m[1]) and [1, m[0]] or [m[0]]
#
# res = f(a)
#
# for i in range(1, len(a)):
#     if res[-1] != a[i]:
#         res += f(a[i:])
#         continue
#     res[-2] += 1
#
# print(*res, sep='')

# def pack(a, i=0):
#     res = []
#     m = re.search(r'^(\w)\1*', a)
#     if m:
#         beg, end = m.span()
#         res = [end, a[0]] if end != 1 else [a[0]]
#         res += pack(a[end:], i+end)
#     return res
#
# print(*pack(input()),sep='')

# print(*[str(len(i[0]))+i[1] if len(i[0]) > 1 else i[1] for i in re.findall(r'((\w)\2*)', a)], sep='')


# import json
# import requests
#
# # num = [31, 999, 1024, 502]
# api_url = 'http://numbersapi.com'
# type = 'math?json'
# with open('dataset_24476_3.txt','r') as f:
#     for num in f:
#         resp = requests.get("/".join([api_url, num.rstrip(), type]))
#         print('Interesting' if json.loads(resp.text)['found'] else 'Boring')

# for i in num:
#     resp = requests.get("/".join([api_url, str(i), type]))
#     print('Interesting' if json.loads(resp.text)['found'] else 'Boring')


# for i in num:
#     print('Boring' if 'uninteresting' in requests.get("/".join([api_url, str(i), type])).text else 'Interesting')

# data = json.loads(input())
# def childs(parent):
#     c = {parent}
#     for child in data:
#         if parent in child['parents']:
#             c |= childs(child['name'])
#     return c
#
#
# [print("{} : {}".format(p['name'], len(childs(p['name'])))) for p in sorted(data, key=lambda i: i['name'])]

# https://stepik.org/lesson/24473/step/2?thread=solutions&unit=6777
# https://stepik.org/media/attachments/lesson/24473/Crimes.csv
# import csv
# import sqlite3
# from sqlite3 import Error
# from datetime import datetime
#
# col = """ID,CaseNumber,Date,Block,IUCR,PrimaryType,Description,LocationDescription,
#     Arrest,Domestic,Beat,District,Ward,CommunityArea,FBICode"""
# conn = None
# try:
#     conn = sqlite3.connect(':memory:')
#     c = conn.cursor()
#     c.execute("CREATE TABLE crimes ({})".format(col))
#     with open('Crimes.csv', 'r') as f:
#         dr = csv.DictReader(f)
#         data = []
#         for i in dr:
#             data += [(i['ID'],
#                       i['Case Number'],
#                       str(datetime.strptime(i['Date'], '%m/%d/%Y %I:%M:%S %p')),
#                       i['Block'],
#                       i['IUCR'],
#                       i['Primary Type'],
#                       i['Description'],
#                       i['Location Description'],
#                       i['Arrest'],
#                       i['Domestic'],
#                       i['Beat'],
#                       i['District'],
#                       i['Ward'],
#                       i['Community Area'],
#                       i['FBI Code'])]
#
#     c.executemany("INSERT INTO crimes VALUES ({})".format(("?,?,?," * 5)[:-1]), data)
#     conn.commit()
#     c.execute("""SELECT Count(*) c,PrimaryType
#         FROM crimes WHERE Date BETWEEN '2015-01-01' AND '2015-12-31'
#         GROUP BY PrimaryType ORDER BY c DESC""")
#     rows = c.fetchall()
#     [print(*row) for row in rows]
#
# except Error as e:
#     print(e)
# finally:
#     if conn:
#         conn.close()

# a = '09/27/2002 05:15:00 AM'
# dt = datetime.strptime(a, '%m/%d/%Y %I:%M:%S %p')
# print(str(dt))

#
# url = 'http://pastebin.com/raw/7543p0ns'
# # url = input()
# regex = r'(?<=<a)(?:[\s\w"\'=])*(?:href="|\')(?:https?|ftp)?(?:\:\/\/)?(\w[\w\.-]*)'
# urlist = {m for m in re.findall(regex, requests.get(url).text, flags=re.IGNORECASE)}
# print(*sorted(urlist), sep='\n')


# (https?|ftp):\/\/[^\s\/$.?#].[^\s>"]*
# http[\w:\/\.]+
# def getref(url):
#     resp = requests.get(url)
#     if resp.status_code == 200:
#         return [match.group() for match in re.finditer(r'<a[^>]*?href="(.*?)"[^>]*?>', resp.text)]
#     else:
#         return []
#
# urlA = '<a href="https://stepic.org/media/attachments/lesson/24472/sample0.html">0</a>\n' \
#        '<a href="https://stepic.org/media/attachments/lesson/24472/sample1.html">1</a>\n' \
#        '<a href="https://stepic.org/media/attachments/lesson/24472/sample2.html">2</a>'
#
# for url in re.findall(r'<a[^>]*?href="(.*?)"[^>]*?>', urlA):
#     print(url)

# for match in m:
#     print(match.group())

# urlA, urlB = [input() for _ in range(2)]
#
# urlist = getref(urlA)
# res = 'No'
# for url in urlist:
#     if urlB in getref(url):
#         res = 'Yes'
#         break
# print(res)

# import sys
#
# [print(re.sub(r'((\w)\2+)', r'\2', line.rstrip())) for line in sys.stdin]

# s, t = [input() for _ in range(2)]
# i = c = 0
#
# while t in s[i:]:
#     i = s.find(t, i)
#     i += 1
#     c += 1
# print(c)

# s, a, b = [input() for _ in range(3)]
# c = 0 if a != b or a not in s else 1001
# while c <= 1000:
#     if a not in s:
#         break
#     s = s.replace(a, b)
#     c += 1
# print('Impossible' if c > 1000 else c)


# def mod_checker(x, mod=0):
#     return lambda y: y % x == mod

# mod_checker = lambda x, mod=0: lambda y: y % x == mod

# import os
# with open('outf.txt','w') as outf:
#     outf.write('\n'.join(sorted({curdir for curdir, subdirs, files in os.walk('main') for f in files if f.endswith('.py')})))
# print(*sorted({curdir for curdir, subdirs, files in os.walk('main') for f in files if f.endswith('.py')}), sep='\n')


# class NonPositiveError(Exception):
#     def __init__(self, *args, **kwargs):
#         Exception.__init__(self, *args, **kwargs)
#
#
# class PositiveList(list):
#     def append(self, x):
#         if x > 0:
#             super(PositiveList, self).append(x)
#         else:
#             raise NonPositiveError()

# try:
#     foo()
# except ZeroDivisionError:
#     print("ZeroDivisionError")
# except ArithmeticError:
#     print("ArithmeticError")
# except AssertionError:
#     print("AssertionError")

# ext = {}
# for i in range(int(input())):
#     e = input().split(':')
#     ext[e[0].strip()] = e[1].split() if len(e) > 1 else []
#
#
# def has_parent(parent, child):
#     # res = False
#     # if parent == child or parent in ext[child]:
#     #     return True
#     # for subparent in ext[child]:
#     #     if has_parent(parent, subparent):
#     #         res = True
#     #         break
#     # return res
#     return parent == child or any(map(lambda p: has_parent(parent, p), ext[child]))
#
#
# [print('Yes' if has_parent(*input().split()) else 'No') for _ in range(int(input()))]

# print(ext)
# print(has_parent('A','D'))


# class ExtendedStack(list):
#     def append(self, val):
#         super(ExtendedStack, self).append(val)
#
#     def pop(self):
#         return super(ExtendedStack, self).pop()
#
#     def sum(self):
#         """операция сложения"""
#         self.append(self.pop() + self.pop())
#
#     def sub(self):
#         """операция вычитания"""
#         self.append(self.pop() - self.pop())
#
#     def mul(self):
#         """операция умножения"""
#         self.append(self.pop() * self.pop())
#
#     def div(self):
#         """операция целочисленного деления"""
#         self.append(self.pop() // self.pop())

# import time
#
# class Loggable:
#     def log(self, msg):
#         print(str(time.ctime()) + ": " + str(msg))
#
# class LoggableList(list,Loggable):
#     def append(self, val):
#         super(LoggableList,self).append(val)
#         self.log(val)

# class Buffer:
#     def __init__(self):
#         self.buffer = []
#
#     def add(self, *a):
#         for i in a:
#             self.buffer.append(i)
#             if len(self.buffer) == 5:
#                 print(sum(self.buffer))
#                 self.buffer.clear()
#
#     def get_current_part(self):
#         return self.buffer


# class MoneyBox:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.amount = 0
#
#     def can_add(self, v):
#         return self.amount + v <= self.capacity
#
#     def add(self, v):
#         self.amount += v

# def c(n, k):
#     if k == 0: return 1
#     if k > n: return 0
#     return c(n - 1, k) + c(n - 1, k - 1)
#
# n, k = map(int, input().split())
# print(c(n ,k))
# def closest_mod_5(x):
#     if x % 5 == 0:
#         return x
#     for y in range(x, x*5):
#         if y % 5 == 0:
#             return y

# objects = [1,2,3,1,True]
# print(len(set(map(id))))


# print(sum([int(input()) for i in range(int(input()))]))

# fib = lambda x : 1 if x <= 2 else fib(x - 1) + fib(x - 2)
# print(fib(31))

# import requests
# import re
#
# def getfile(url):
#     return requests.get(url).text
# url ='https://stepic.org/media/attachments/course67/3.6.3/'
#
# with open('dataset_3378_3.txt', 'r') as linkTo:
#     text = linkTo.readline().strip()
#
# text = getfile(text)
#
# while not re.match(r'^We\s.*', text):
#     text = getfile(url+text)
#
# with open('uotput.txt','w') as outf:
#     outf.write(text)

# with open('dataset_3378_2.txt', 'r') as linkTo:
#     r = requests.get(linkTo.readline().strip())
# print(len(r.text.splitlines()))

# s = [0] * 3
# n = 0  # счетчик учеников
# outf = open("output.txt", "w")
# with open('dataset_3363_4.txt', 'r') as dataset:
#     for line in dataset:
#         a = line.split(';')
#         notes = [int(i) for i in a[1:]]  # оценки за каждый предмет одного ученика
#         for i in range(3):
#             s[i] += notes[i]  # накопление суммы ценок всех учеников
#         outf.write(str(sum(notes) / 3) + '\n')  # запись среднего балла для каждого
#         n += 1
#     outf.write(" ".join([str(i / n) for i in s]))  # средний балл по каждому предмету
# outf.close()

# with open("dataset_3363_3.txt", "r") as dataset:
#     a = dataset.read().lower().split()
# stat = {i: a.count(i) for i in set(a)}
# mostFreq = {k: v for k, v in stat.items() if v == max(stat.values())}
# for key in sorted(mostFreq):
#     print(key, mostFreq[key])

# print(max(uns.values()))
# print(sortd.items())
# print(sorted(uns.items(), key=lambda item: item[1], reverse=True))

# import re
# a = 'a3b4c2e10b1'
# inf = open("dataset_3363_2.txt", "r")
# outf = open("output.txt", "w")
# a = inf.readline()
# outf.write("".join([match.group()[:1] * int(match.group()[1:]) for match in re.finditer(r"[a-zA-Z]\d+", a)]))
# outf.close()
# inf.close()

# a = [int(input()) for i in range(input())]
# b = {x: f(x) for x in set(a)}
# for x in a:
#     print(x, b[x])

# n = int(input())
# a = {}
# while n > 0:
#     x = int(input())
#     if x not in a:
#         a[x] = f(x)
#     print(a[x])
#     n -= 1

# warAndPiece = 'a aa abC aa ac abc bcd a'.lower().split() # input().lower().split()
# a = {word: warAndPiece.count(word) for word in warAndPiece}
# for w in warAndPiece:
#     if w.lower() not in a:
#         a[w.lower()] = 1
#     else:
#         a[w.lower()] += 1
# for key,val in a.items():
#     print(key, val)
# print(*a.items(), sep='\n')
# def update_dictionary(d, key, value):
#     if key in d.keys():
#         d[key] += [value]
#     elif 2 * key in d.keys():
#         d[2 * key] += [value]
#     else:
#         d[2 * key] = [value]

# a, b = int(input()), int(input())
# s = list(i for i in range(a, b+1) if i % 3 == 0)
# print(sum(s)/len(s))
# s = 'abcdefghijk'
# print(s[-1:-10:-2]) # :-2

# (x for x in xyz if x not in a)
# int(input(),int(input()))
# a, b = int(input(),int(input()))
# c, d = int(input(),int(input()))
# for i in range(a-1, b+1):
#     print('' if i < a else i, end='\t')
#     for j in range(c, d+1):
#         print(j if i < a else i*j, end='\t' if j < d else '\n')

# while True:
#     n = int(input())
#     if n < 10:
#         continue
#     elif n > 100:
#         break
#     else:
#         print(n)

# a = int(input())
# b = int(input())
# pices = min(a, b)
# m = pices
# while pices % a != 0 or pices % b != 0:
#     pices += m
# print(pices)
# sum = 0
# while True:
#     n = int(input())
#     if n == 0:
#         break
#     sum += n
# print(sum)
# ticketNr = '090234' # input()
# firstSum, secondSum = 0, 0
# for i in range(ticketNr.__len__()):
#     secondSum += int(ticketNr.__getitem__(i))
#     if i == ticketNr.__len__()//2-1:
#         firstSum, secondSum = secondSum, 0
# print('Счастливый' if firstSum == secondSum else 'Обычный')

# def triangle():
#     a = int(input())
#     b = int(input())
#     c = int(input())
#     p = (a+b+c)/2
#     print((p*(p-a)*(p-b)*(p-c))**0.5)
#
#
# def rect():
#     print(int(input())*int(input()))
#
#
# def circle():
#     print(3.14*int(input())**2)
#
#
# def shapes(shape):
#     return {
#         'треугольник': triangle,
#         'прямоугольник': rect,
#         'круг': circle
#     }.get(shape, 'error')
#
#
# shapes(input())()
