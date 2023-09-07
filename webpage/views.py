import os
from django.http import HttpResponse, HttpRequest, FileResponse, StreamingHttpResponse, JsonResponse, HttpResponseNotFound
from django.shortcuts import render
from django.conf import settings
import cv2
from .cchecker import cchecker
import uuid
import json
import base64

image_lib = {}

def index_page(req: HttpRequest):
    if req.method == 'GET':  
        p = req.path
        if p == '/':
            return render(req, 'index.html')
        if os.path.isfile(f'{settings.TEMPLATES[0]["DIRS"][0]}{p}'):
            return render(req, p[1:])
    return HttpResponseNotFound(req)

def upload_image(req: HttpRequest):
    c = cchecker(req.FILES['file'].read())
    if not c.get_init():
        return JsonResponse({'status': 0, 'des': 'Cannot find checker'})
    c.run()
    c.infer()
    ret, buffer = cv2.imencode('.jpg', cv2.cvtColor(c.get_output(), cv2.COLOR_RGB2BGR))
    buffer = buffer.tobytes()
    u = str(uuid.uuid1())
    image_lib[u] = buffer
    return JsonResponse({'status': 1, 'des': 'OK', 'file': u, 'curve': c.get_draw_plot_srgb()})

def download_image(req:HttpRequest):
    a = json.loads(req.body)
    t = image_lib[a['name']]
    del image_lib[a['name']]
    return HttpResponse(t, content_type='blob')

def detect_pattern(req: HttpRequest):
    return JsonResponse({'x':0, 'y':0})