import pymysql
import pandas as pd 

class AuDDB():
    def __init__(self, args):
        self.args = args 

    def connect(self):
        self.conn = pymysql.connect(host=self.args.host, user=self.args.user, \
            password=self.args.password, db=self.args.db, charset='utf8')
        self.curs = self.conn.cursor()

    def execute_sql(self, sql):
        self.curs.execute(sql)
        return self.curs.fetchall() 

    def update_sql(self, sql):
        self.curs.execute(sql)
        self.curs.fetchall() 

    def get_aud_log(self):
        sql = 'SELECT * FROM AuD_log;'
        self.aud_log = self.execute_sql(sql)

    def table_to_csv(self, tb):
        columns=['idx', 'user', 'input_text', 'bws_label', 'dsm_label', 'tokens', 'datetime']
        self.aud_df = pd.DataFrame(tb, columns=columns)
        
    def save_aud_log(self, user, input_text, bws_label, dsm_label, tokens, datetime):
        '''
        save user, input_text, bws_label, dsm_label, tokens, datetime to AuD_log
        ''' 
        sql = "INSERT INTO AuD_log(user, input_text, bws_label, dsm_label, tokens, datetime)\
              VALUES('{0}','{1}','{2}', '{3}', '{4}', '{5}')".format(user, input_text, bws_label, dsm_label, tokens, datetime)
        
        self.execute_sql(sql)
        self.conn.commit()