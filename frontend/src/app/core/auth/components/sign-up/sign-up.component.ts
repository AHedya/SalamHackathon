import { Component, inject } from '@angular/core';
import { AbstractControl, FormBuilder, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { log } from 'node:console';

@Component({
  selector: 'app-sign-up',
  imports: [ReactiveFormsModule],
  templateUrl: './sign-up.component.html',
  styleUrl: './sign-up.component.css'
})
export class SignUpComponent {
  private readonly router = inject(Router);
  private readonly _formBuilder = inject(FormBuilder)
  isLoading: boolean = false;
  messageError: string = '';
  messageSuccess: string = '';

  registerForm:FormGroup= this._formBuilder.group({
    name:[null,[ Validators.required,Validators.minLength(3),Validators.maxLength(30),]],
    email:[null, [ Validators.required, Validators.pattern(/^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/),]],
    password:[null,[ Validators.required,  Validators.pattern(/^\w{6,}$/),]],
    rePassword:[null,[Validators.required]],
    phone:[null,[Validators.required, Validators.pattern(/^01[0125][0-9]{8}$/),]],
  },{ validators: this.confirmPassword})


  confirmPassword(group: AbstractControl) {
    const password = group.get('password')?.value;
    const rePassword = group.get('rePassword')?.value;

    return password === rePassword ? null : { mismatch: true };
  }

  submitForm(): void {
    if (this.registerForm.valid) {
      this.isLoading = true; 
      this.messageError = ''; 
      this.messageSuccess = 'تم التسجيل بنجاح!';
  
      setTimeout(() => {
        this.isLoading = false; 
        this.router.navigate(['/signIn']); 
      }, 1500);
  
    } else {
      this.registerForm.markAllAsTouched();
      this.messageError = 'يرجى ملء جميع الحقول بشكل صحيح.';
    }
  }
  
}
