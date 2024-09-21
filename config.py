from flask import Flask
from flask_cors import CORS
from flasgger import Swagger, LazyString
from json import JSONEncoder as BaseJSONEncoder

class CustomJSONEncoder(BaseJSONEncoder):
    def default(self, obj):
        if isinstance(obj, LazyString):
            return str(obj)
        return super().default(obj)

def create_app():
    app = Flask(__name__)
    app.json_encoder = CustomJSONEncoder  

    CORS(app)

    swagger_template = dict(
        info={
            'title': 'Sentiment Analysis API V1.0.0',
            'version': '1',
            'description': 'Data Science Binar Academy Platinum Challenge Oleh Kelompok 2: Badrudin, Elfilia, Asip'
        },
        host='localhost:5000'
    )

    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": 'api/docs',
                "route": '/api/docs'
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs/"
    }

    swagger = Swagger(app, template=swagger_template, config=swagger_config)

    return app
