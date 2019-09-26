#在这里完成对数据库的操作
import pymysql
import base64
db = pymysql.connect(host='localhost', user='root', password='root', db='ai_mirror', port=3306)
cur = db.cursor()

def store(pic):
    try:
        sql_insert = "insert into image(pic) values ('{}')".format(pic)
        print(sql_insert)
        cur.execute(sql_insert)
        db.commit()
    except Exception as e:
        print(e)
        db.rollback()

if __name__=='__main__':
    s='lyplalala!'.encode("utf-8")
    store(bytes.decode(base64.b64encode(s)))