
# %%
from functions import (call_driver, send_focus_key,
                       get_text_and_links, date_transform)
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


# instantiate the drive with predefined option
driver = call_driver()

# %%
# the words to focus on the serach in google
focus_key = 'Lula economia dolar'

# send the focus_key to drive search box
send_focus_key(driver, focus_key=focus_key)

headlines, texts, links = get_text_and_links(driver, pg_num=20)

#%%

df0 = pd.concat([headlines, texts, links], axis=1)
df0 = df0.dropna()
df0['texts'] = df0['texts'].apply(lambda x: x.split('\n')[0])
df0['date'] = [x.split('—')[0] if '—' in x else np.nan for x in df0['headlines']]
df0['headlines_1'] = [x.split('—')[1] if '—' in x else np.nan for x in df0['headlines']]
df0['date'] = df0['date'].apply(date_transform)
df0['headlines_0'] = [x.split('—')[0] if '—' in x else np.nan for x in df0['headlines']]
df0

# %%
df0.to_csv('news_df.csv')


# %%
