import { Component} from '@angular/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-splash-screen',
  imports: [],
  templateUrl: './splash-screen.component.html',
  styleUrl: './splash-screen.component.css',
})
export class SplashScreenComponent {
  constructor(private router: Router) { }

  ngOnInit(): void {
    setTimeout(() => {
      this.router.navigate(['/welcome']);
    }, 3000);
  }
}