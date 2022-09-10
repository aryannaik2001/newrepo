// 1. 
//import java.util.*;

// public class functions{
//     public static void printMyName(String name){
//         System.out.println(name);
//         return;
//     }

//     public static void main(String args[]){
//         Scanner sc = new Scanner(System.in);
//         String name = sc.next();

//         printMyName(name);
//     }
//
// 2.
// import java.util.*;
// public class functions{

//     public static int Sum(int a, int b){
//         int c = a+b;
//         return c;
//     }

//     public static void main(String args[]){

//         Scanner sc = new Scanner(System.in);
//         int a = sc.nextInt();
//         int b = sc.nextInt();
//         int c = Sum(a,b);
//         System.out.println(c);

//     }
// }

// 3.
// import java.util.*;
// public class functions{
//     public static int multiply(int x, int y){
//         return x*y;
//     }
//     public static void main(String args[]){
//         Scanner sc = new Scanner(System.in);
//         int a = sc.nextInt(); int b = sc.nextInt();
//         System.out.println(multiply(a,b));
//     }
// }

import java.util.*;

public class functions{
    public static void Fact(int x){
        int Product = 1;
        if(x<0){
            System.out.println("Invalid number");
        }
        else if(x==0){
            System.out.println("1");
        }
        else{
        for(int i=1;i<=x;i++){
            Product = Product*i;
        }
        System.out.println(Product);}
    }
    public static void main(String args[]){
        Scanner sc = new Scanner(System.in);
        int a = sc.nextInt();

        Fact(a);
    }
}