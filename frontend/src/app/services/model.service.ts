import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ModelService {
  private apiUrl = 'http://localhost:8000'; // Ajusta si usas otro host/puerto

  constructor(private http: HttpClient) {}

  setModel(model: '1_layer' | '2_layers'): Observable<{ active_model: string }> {
    const formData = new FormData();
    formData.append('model', model);
    return this.http.post<{ active_model: string }>(`${this.apiUrl}/set_model`, formData);
  }
}
