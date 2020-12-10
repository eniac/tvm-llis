import sys, requests, json, numpy as np

model_name = sys.argv[1]
input_dim = int(sys.argv[2])
input_shape = [int(x) for x in sys.argv[3:input_dim+3]]
input_output_path = sys.argv[-1]

input_np = np.random.randn(1, *input_shape).astype(np.float32)
input_list = input_np.tolist()
print('input_list:', input_list)

input_str = json.dumps({
    "instances": input_list
})

with open(input_output_path, 'w') as f:
    f.write(input_str)

url = 'http://localhost:8501/v1/models/{}:predict'.format(model_name)

headers = {"Content-type": "application/json"}
print(requests.post(url, headers=headers, data=json.dumps({
    "instances": input_list})).json())

