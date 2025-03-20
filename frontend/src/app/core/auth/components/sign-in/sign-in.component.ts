import { Component, inject } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink} from '@angular/router';

@Component({
  selector: 'app-sign-in',
  imports: [ReactiveFormsModule,RouterLink],
  templateUrl: './sign-in.component.html',
  styleUrl: './sign-in.component.css'
})
export class SignInComponent {

  private readonly router = inject(Router);
  isLoading: boolean = false;
  messageError:string=''
  messageSuccess:string=''
  loginForm: FormGroup = new FormGroup(
    {
      email: new FormControl(null, [
        Validators.required,
        Validators.pattern(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/),
      ]),
      password: new FormControl(null, [
        Validators.required,
        Validators.pattern(/^\w{6,}$/),
      ]),
    }
  );


  submitForm(): void {
    if (this.loginForm.valid) {
      this.isLoading = true; 
      this.messageError = ''; 
      this.messageSuccess = 'تم التسجيل بنجاح!';
  
      setTimeout(() => {
        this.isLoading = false; 
        this.router.navigate(['/home']); 
      }, 1500);
  
    } else {
      this.loginForm.markAllAsTouched();
      this.messageError = 'يرجى ملء جميع الحقول بشكل صحيح.';
    }
  }


}
