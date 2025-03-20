import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class UploadSoundService {

  private apiUrl = 'http://127.0.0.1:8000/TTS/';

  constructor(private http: HttpClient) {}

  getAudio(text: string): Observable<Blob> {
    const url = `${this.apiUrl}?text=${encodeURIComponent(text)}`; 
    return this.http.get(url, { responseType: 'blob' });
  }

}
