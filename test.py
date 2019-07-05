import urllib.parse
import re

ascii_symbols = r'[ -/:-@[-`{-~]' # asciiコード中の記号(スペース含む)とマッチする正規表現
ascii_control_chars = r'[\x00-\x1f]|\x7f' # asciiコード中の制御文字とマッチする正規表現
reg = re.compile('({}|{})'.format(ascii_symbols, ascii_control_chars))

str = 'id=2&nombre=Jam%F3n+Ib%E9rico&precio=85&cantidad=%27%3B+DROP+TABLE+usuarios%3B+SELECT+*+FROM+datos+WHERE+nombre+LIKE+%27%25&B1=A%F1adir+al+carrito'
str = urllib.parse.unquote_plus(str)

arr = reg.split(str)
print(arr)