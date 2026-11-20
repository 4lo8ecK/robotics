# python module myui for python labs in university
# COLORS

clear   = 0 
bold    = 1 
dim     = 2
italic  = 3
underline   = 4
blink   = 5
select  = 7
cross   = 9
double_underline = 21
thin_underline = 52

black   = 30 
red     = 31 
green   = 32 
yellow  = 33 
blue    = 34 
purple  = 35 
cyan    = 36 
grey    = 37

bg_black   = 40 
bg_red     = 41 
bg_green   = 42 
bg_yellow  = 43 
bg_blue    = 44 
bg_purple  = 45 
bg_cyan    = 46 
bg_grey    = 47

def style(*args: int) -> str:
    
    res = '\x1b[0'
    for i in args:
        if not isinstance(i, int):
            continue
        res += ';' + str(i)
        pass
    res += 'm'
    return res

if __name__ == "__main__":
    print(f"{style(bold, red)}test{style(0)}")
