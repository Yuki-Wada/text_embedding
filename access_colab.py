import time
import datetime
import webbrowser

colab_url = 'https://colab.research.google.com/drive/17IGiNOGbtT6BA-QUSA4A2f_xw6xC5uZi#scrollTo=GufJVmPhU8kP'

for i in range(12):
    webbrowser.open(colab_url)
    print(i, datetime.datetime.today())
    time.sleep(60 * 60)
