# !/user/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Henry'

from app import app
from flask_script import Manager

# manage = Manager(app)

if __name__ == '__main__':
    # manage.run()
    app.run()
    # app.run(port=8080)
