
import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { UploadComponent } from './upload/upload.component';
import { SearchComponent } from './search/search.component';
import { SearchResultsComponent } from './results/search-results.component';
import { ModelService } from './services/model.service';

@Component({
  selector: 'app-root',
  imports: [ UploadComponent, SearchComponent, SearchResultsComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('Sistema de búsqueda de imágenes por similitud');
  activeModel: '1_layer' | '2_layers' = '1_layer';

  constructor(private modelService: ModelService) {}

  setModel(model: '1_layer' | '2_layers') {
    this.modelService.setModel(model).subscribe({
      next: (resp) => {
        this.activeModel = resp.active_model as '1_layer' | '2_layers';
      },
      error: (err) => {
        alert('Error cambiando modelo: ' + (err?.error?.detail || err.message));
      }
    });
  }
}
