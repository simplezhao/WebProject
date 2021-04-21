from flask import Flask
# 1.创建app
from .views import init_route

def create_app():
	app = Flask(__name__)
	# 2. 想将views导入，但是这里导入的话 ，views还是要导入app 因此采用函数调用的方式，这种方式也称为懒加载
	init_route(app)
	return app