AUTHOR = 'bo'
SITENAME = 'bo.blog'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'Europe/Berlin'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (('Pelican', 'https://getpelican.com/'),
         ('Python.org', 'https://www.python.org/'),
         ('Jinja2', 'https://palletsprojects.com/p/jinja/'),)

# Social widget
SOCIAL = (('Github', 'https://github.com/bwanglzu'),
          ('Twitter', 'https://twitter.com/bo_wangbo'),)

DEFAULT_PAGINATION = False

THEME = "/Users/bo/Documents/work2/bwanglzu.github.io/venv/lib/python3.9/site-packages/pelican/themes/martin-pelican"

PLUGIN_PATHS=['./plugins']
PLUGINS = ['render_math']

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True