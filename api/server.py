# server.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/poem/invoke', methods=['POST'])
def invoke_poem():
    data = request.get_json()
    response = {
        "message": "Poem generated successfully!",
        "input": data
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='localhost', port=8080)
