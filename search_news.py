
# %%
from functions import (call_driver, send_focus_key,
                       get_text_and_links, date_transform)
import pandas as pd

# instantiate the drive with predefined option
driver = call_driver()

# %%
# the words to focus on the serach in google
focus_key = 'Lula economia dolar'

# send the focus_key to drive search box
send_focus_key(driver, focus_key=focus_key)

headlines, texts, links = get_text_and_links(driver, pg_num=1)

# %%
date = texts.str.split('—').str[0].str.replace(',', '')
date = date.apply(date_transform)
date
# date = date.str.replace(',','')


# %%
pd.concat(headlines, texts, links).to_csv('news_df.csv')
—
