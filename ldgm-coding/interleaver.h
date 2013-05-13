#ifndef INTERLEAVER_H
#define INTERLEAVER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 *This struct is used for return two integer number.
 *Integer position is number where is copied symbol or part of symbol.
 *Integer lengthOfSymbol is length of symbol which is in packet (it can be only part of symbol in packet)
 */
/*
 * Tato struktura je zde pro vraceni hodnot, pri dekodovani.
 * Struktura Symbol uchovavá pozici v buferu, kam se má symbol kopírovat
 * a velikost symbolu, která se má kopirovat.
 */

typedef struct symbol{
    int position;
    int lengOfSymbol;
}Symbol;


/**
 *This function shuffle pointers of symbols.
 *@param *array is pointer to array of pointers
 *@param *shuffledArray is pointer to array. Here are saved shuffled symbol's pointers.
 *@param koefQuadratic is an integer of quadratic koeficient which is used in interleaving
 *@param koefLinear is an integer of linear koeficient which is used in interleaving
 *@param koefCyclicShuffle is an integer of cyclic shuffle koeficient which is used in interleaving
 *@param numberOfPieces is a number of symbols per one frame
 */
  /*Tato funkce promicha dilky dle zadanych koeficientu.
  Metoda pracuje s pointery na dve stejne velka pole pointeru.
  Po vypoctu ukladame data na i-ty index z nami vypocteneho indexu, abychom mohli zpetne rict, kam se data maji kopirovat.
  @param *array je pointer na pole, ktere se ma promichat
  @param *shuffledArray je pointer na pole, kam se maji ulozit promihcane dilky
  @param koefQuadratic udava hodnotu kvadratickeho koeficientu pri vypoctu indexu
  @param koefLinear udava hodnotu linearniho koeficientu pri vypoctu indexu
  @param koefCyclicShuffle udava hodnotu cyklickeho posunu pri vypoctu indexu
  @param numberOfPieces udava pocet dilku, na ktere je rozdelen frame
  */
void shuffle(char **array, char **shuffledArray, int koefQuadratic, int koefLinear, int koefCyclicShuffle, int numberOfPieces);


/**
 *Function counts index......
 *@param koefQuadratic is an integer of quadratic koeficient which is used in interleaving
 *@param koefLinear is an integer of linear koeficient which is used in interleaving
 *@param koefCyclicShuffle is an integer of cyclic shuffle koeficient which is used in interleaving
 *@param numberOfPieces is a number of symbols per one frame
 *@param oldIndex is an integer number of index in shuffled array
 *
 *@return index of array before shuffling
 */
/*
  Vypocet indexu povodniho pole z predem daneho predpisu a indexu v promichanem poli.
  @param koefQuadratic udava hodnotu kvadratickeho koeficientu pri vypoctu indexu.
  @param koefLinear udava hodnotu linearniho koeficientu pri vypoctu indexu.
  @param koefCyclicShuffle udava hodnotu cyklickeho posunu pri vypoctu indexu.
  @param numberOfPieces udava pocet dilku, na ktere je rozdelen frame.
  @param oldIndex udava hodnotu indexu v promichanem poli.

  @return Index v puvodnim poli.
  */
int countIndex(int koefQuadratic, int koefLinear, int koefCyclicShuffle, int numberOfPieces, int oldIndex);


/*!
 *This function from packet deshuffle symbols to original index.
 *Function compute original index of symbol and length of data of symbol (not evry symbols is whole in one packet).
 *This information (original index and length of data) are copied to structure which is saved in array of structure.
 *@param offset is an integer number from incoming packet
 *@param lengthOfPacket is an integer of length of incoming packet
 *@param koefQuadratic is an integer of quadratic koeficient which is used in interleaving
 *@param koefLinear is an integer of linear koeficient which is used in interleaving
 *@param koefCyclicShuffle is an integer of cyclic shuffle koeficient which is used in interleaving
 *@param numberOfPieces is a number of symbols per one frame
 *@param lengthOfSymbol is an integer number of length of one symbol
 *@param *arrayOfSymbols is pointer to array where we save structure named Symbol
 *
 *@return integer number of symbols which are in one incoming packet.
 */
  /*Tato funkce z prijateho packetu zjisti puvodni index symbolu a vraci velikost symbolu obsazeneho v packetu a pozici, kam tento symbo kopirovat.
  @param offset udava hodnotu offsetu prichoziho packetu.
  @param lengthOfPacket udava velikost prichoziho packetu.
  @param koefQuadratic udava hodnotu kvadratickeho koeficientu pri vypoctu indexu.
  @param koefLinear udava hodnotu linearniho koeficientu pri vypoctu indexu.
  @param koefCyclicShuffle udava hodnotu cyklickeho posunu pri vypoctu indexu.
  @param numberOfPieces udava pocet dilku, na ktere je rozdelen frame.
  @param lengthOfSymbol udava velikost jednohu symbolu, ktera je pevne dana.
  @param *arrayOfSymbols uchovava strukturu symbol, ktera obsahuje

  @return počet alokovaných symbolů v poli
  */
int deshuffle(int offset, int lengthOfPacket, int koefQuadratic, int koefLinear, int koefCyclicShuffle, int numberOfPieces,int lengthOfSymbol,Symbol *arrayOfSymbols);

#ifdef __cplusplus
}
#endif

#endif // INTERLEAVER_H
