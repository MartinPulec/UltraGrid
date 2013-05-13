#include <stdio.h>
#include <string.h>
#include "interleaver.h"

void shuffle(char **array, char **shuffledArray, int koefQuadratic, int koefLinear, int koefCyclicShuffle, int numberOfPieces){
    for(int i=0; i<numberOfPieces;i++){
        int result = countIndex(koefQuadratic,koefLinear,koefCyclicShuffle,numberOfPieces,i);
        shuffledArray[i]=array[result];
    }
}

int countIndex(int koefQuadratic, int koefLinear, int koefCyclicShuffle, int numberOfPieces, int oldIndex){

    int newIndex =((koefQuadratic*((oldIndex*oldIndex)%numberOfPieces)%numberOfPieces + (koefLinear*oldIndex)%numberOfPieces +koefCyclicShuffle)%numberOfPieces);
    return newIndex;
}

int deshuffle(int offset, int lengthOfPacket, int koefQuadratic, int koefLinear, int koefCyclicShuffle, int numberOfPieces, int lengthOfSymbol, Symbol *arrayOfSymbols){
    int endOfPacket;      // @variable endOfPacket ukazuje na konec paketu
    if(offset+lengthOfPacket<numberOfPieces*lengthOfSymbol){
        endOfPacket = offset+lengthOfPacket;
    }else{
        endOfPacket=numberOfPieces*lengthOfSymbol;
    }

    int tempOffset=offset;
    int tempIndex;      // @variable tempIndex je pomocna pro vypocet prubezneho indexu
    int originalIndex;  // @variable originalIndex promena uchovavajici hodnotu puvodniho indexu
    int lengthOfData;   // @variable lengthOfData promena udrzuje velikost kopirovanych dat
    unsigned position;       // @variable position uchovava hodnotu ukazujici na misto, kam se maji kopirovat data
    Symbol data;
    int count=0;        // @variable udava kolik symbolu bylo v paketu

    while(tempOffset<endOfPacket){
        tempIndex=tempOffset/lengthOfSymbol;    // @variable zjisti index prichoziho symbolu
        originalIndex=countIndex(koefQuadratic,koefLinear,koefCyclicShuffle,numberOfPieces,tempIndex);  // @value spocte puvodni index

        if(tempOffset+lengthOfSymbol>endOfPacket){          // rozhodne, zda je cely symbol v packetu nebo jen castecne a provede potrebne kroky vypoctu
            if(tempOffset%lengthOfSymbol==0){              // rozhodne, zda se zacina na zacatku symbolu ci ne a dle toho spocte velikost kopirovanych dat
                lengthOfData=endOfPacket-tempOffset;        //spocte hodnotu kopirovanych dat
            }else{
                lengthOfData=((tempIndex+1)*lengthOfSymbol)-offset; // @value spocte velikost kopirovanych dat, pokud se cely symbol nevesel do packetu

                if(lengthOfData>lengthOfPacket){        // pokud hodnota kopirovanych presahuje velikos packetu, tak musime kopirovat pouze velikost packetu
                    lengthOfData=lengthOfPacket;
                }
            }
            position = originalIndex*lengthOfSymbol+(tempOffset-tempIndex*lengthOfSymbol);        // @value spocte misto, kam se maji kopirovat data bez posunu
        }else{
            lengthOfData = tempIndex*lengthOfSymbol+lengthOfSymbol-tempOffset;          // @value spocte velikost kopirovanych dat, pro symboly, ktere doplnuji nevesle znaky do predchazejiciho packetu a pro cele symboly v packetu
            position = originalIndex*lengthOfSymbol+(lengthOfSymbol-lengthOfData);     // @value spocte misto, kam se maji kopirovat data s posunem dat o jiz prenesene znaky
        }

        tempOffset=tempOffset+lengthOfData;     // @variable posune offset v bufferu o velikost kopirovanych dat
        data.position=position;                 // ulozi pozici prvave prichoziho symbolu
        data.lengOfSymbol=lengthOfData;         // ulozi velikost prichoziho symbolu
        arrayOfSymbols[count]=data;
        count+=1;

    }
    return count;
}
