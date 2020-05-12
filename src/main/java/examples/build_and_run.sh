#!/bin/sh

export CLASSPATH=.:../lib/mallet.jar:../lib/mallet-deps.jar:../class/
echo CLASSPATH is set to $CLASSPATH

#build all the examples
echo Running: 
echo javac *.java 
if 
javac *.java   
then 
	echo "Success. "
else
	echo Error running build command.  Punting.  
	exit 
fi

echo
echo ================================================== 
echo

# run simple classification example
echo Running: 
echo java SimpleClassificationExample 
java SimpleClassificationExample 

echo
echo ==================================================  
echo

echo Running: 
echo java SimpleSequenceExample 
java SimpleSequenceExample 
