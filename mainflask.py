# Web socket server to talk with the Angular App.
import json

from flask import Flask, request, jsonify
from autogen import config_list_from_json
from flask_cors import CORS, cross_origin
from flask_sock import Sock

# import model
import model_latest


app = Flask(__name__)
sock = Sock(app)
cors = CORS(app)
config_list = config_list_from_json(
    env_or_file="OAI_CONFIG_LIST")


@sock.route('/chatbot', methods=['GET'])
@cross_origin()
def chatbot(ws):
    while True:
        data = ws.receive()
        # print(data)
        # print()
        user_input = json.loads(data)['user_input']  # Assuming input is in JSON format
        print(user_input)
        # chatbot_response = generate_response(user_input)
        # ws.send(jsonify({'chatbot_response': chatbot_response}))
        # ws.send({"source": "test", 'content: chatbot_response})
        ws.send({'source': 'test', 'content': 'Hey this is the response'})

    # return jsonify({'chatbot_response': chatbot_response})


def generate_response(prompt):
    return model_latest.get_cs_help(prompt)
    # default_llm_config = {"config_list": config_list, "seed": 12, "request_timeout": 600}

    # return model.get_cs_help(prompt)


if __name__ == '__main__':
    app.run()